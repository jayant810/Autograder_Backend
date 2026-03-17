import os
import io
import json
import base64
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from PIL import Image
import pdfplumber
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = FastAPI(title="EdulinkX Autograder API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Gemini model for OCR and grading
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Persistence for Answer Keys (Temporary JSON)
ANSWER_KEYS_FILE = "answer_keys.json"


def load_answer_keys():
    if os.path.exists(ANSWER_KEYS_FILE):
        with open(ANSWER_KEYS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_answer_keys(keys):
    with open(ANSWER_KEYS_FILE, 'w') as f:
        json.dump(keys, f, indent=4)


@app.get("/")
async def root():
    return {"status": "Autograder Microservice Online"}


@app.post("/upload-answer-key")
async def upload_answer_key(file: UploadFile = File(...), exam_id: str = Form(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    extracted_data = []
    with pdfplumber.open(file.file) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"

        # Split by "Question" keyword (customize based on your PDF format)
        extracted_data = full_text.split("Question")

    answer_keys = load_answer_keys()
    answer_keys[exam_id] = extracted_data
    save_answer_keys(answer_keys)

    return {"message": "Answer key uploaded and parsed", "exam_id": exam_id}


def perform_ocr_gemini(image_bytes: bytes) -> str:
    """
    Use Gemini Vision to extract handwritten text from an image.
    Returns the extracted text as a string.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")

    model = genai.GenerativeModel(GEMINI_MODEL)

    # Encode image to base64 for the API
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = model.generate_content([
        {
            "mime_type": "image/jpeg",
            "data": image_b64
        },
        """You are a precise OCR engine for handwritten student exam answers.
Extract ALL handwritten text from this image exactly as written.
Rules:
- Output ONLY the extracted text, nothing else.
- Preserve the original spelling (even if there are mistakes).
- Preserve line breaks where they naturally occur.
- Do NOT add any commentary, labels, or formatting.
- If the image is blank or unreadable, respond with: [UNREADABLE]"""
    ])

    return response.text.strip()


def grade_with_gemini_vision(image_bytes: bytes, expected_answer: str, prompt: Optional[str] = None) -> dict:
    """
    Use Gemini Vision to read the student's handwritten answer directly from the image
    and grade it against the expected answer — all in one API call.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")

    model = genai.GenerativeModel(GEMINI_MODEL)

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    grading_prompt = f"""You are an expert exam grader. Look at this student's handwritten answer image and grade it.

Question Context: {prompt if prompt else "Determine from the answer context"}
Expected Answer: {expected_answer}

Instructions:
1. First, read the handwritten text from the image carefully.
2. Compare it with the expected answer.
3. Rate on a scale of 0 to 100 based on accuracy and conceptual understanding.
4. Provide constructive feedback.

Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):
{{"student_text": "what you read from the image", "score": number, "feedback": "your feedback"}}"""

    response = model.generate_content([
        {
            "mime_type": "image/jpeg",
            "data": image_b64
        },
        grading_prompt
    ])

    try:
        result = json.loads(response.text.replace('```json', '').replace('```', '').strip())
    except Exception:
        result = {"student_text": "", "score": 0, "feedback": "Error parsing Gemini response", "raw": response.text}

    return result


@app.post("/grade-image")
async def grade_image(
    file: UploadFile = File(...),
    exam_id: str = Form(...),
    question_idx: int = Form(...),
    method: str = Form("similarity"),  # "similarity" or "gemini"
    gemini_prompt: Optional[str] = Form(None)
):
    # 1. Read Image
    contents = await file.read()

    # 2. Get expected answer from answer key
    answer_keys = load_answer_keys()
    if exam_id not in answer_keys:
        raise HTTPException(status_code=404, detail="Answer key for this exam not found")

    try:
        expected_answer = answer_keys[exam_id][question_idx]
    except IndexError:
        raise HTTPException(status_code=400, detail="Question index out of bounds")

    # 3. Grading Logic
    if method == "similarity":
        # Gemini extracts text → fuzzy match with answer key
        student_text = perform_ocr_gemini(contents)
        score_percentage = fuzz.token_sort_ratio(student_text.lower(), expected_answer.lower())
        final_marks = 100 if score_percentage >= 85 else 0

        return {
            "student_text": student_text,
            "expected_answer": expected_answer,
            "similarity_score": score_percentage,
            "final_marks": final_marks,
            "passed": score_percentage >= 85
        }

    elif method == "gemini":
        # Gemini reads image + grades directly in one shot
        grading_result = grade_with_gemini_vision(contents, expected_answer, gemini_prompt)

        return {
            "student_text": grading_result.get("student_text", ""),
            "grading_result": grading_result
        }

    else:
        raise HTTPException(status_code=400, detail="Invalid grading method. Use 'similarity' or 'gemini'.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

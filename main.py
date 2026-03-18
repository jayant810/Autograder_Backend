import os
import io
import re
import json
import base64
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
from pydantic import BaseModel
from PIL import Image
import pdfplumber
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = FastAPI(title="EdulinkX Autograder API")

# --- API Key Authentication Middleware ---
AUTOGRADER_SECRET_KEY = os.getenv("AUTOGRADER_SECRET_KEY")

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Allow health check without auth
        if request.url.path == "/" or request.url.path == "/docs" or request.url.path == "/openapi.json":
            return await call_next(request)

        # Check API key
        api_key = request.headers.get("X-API-Key")
        if not AUTOGRADER_SECRET_KEY:
            # If no key is configured, allow all (dev mode)
            return await call_next(request)
        if api_key != AUTOGRADER_SECRET_KEY:
            from starlette.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"}
            )
        return await call_next(request)

app.add_middleware(APIKeyMiddleware)

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

# Gemini model for OCR and grading (use flash-lite for predictable non-thinking output)
GEMINI_MODEL = "gemini-2.5-flash-lite"


def extract_json_from_text(text: str) -> dict:
    """
    Robustly extract JSON from Gemini response text.
    Handles markdown code blocks, extra text around JSON, etc.
    """
    cleaned = text.strip()
    
    # 1. Try direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    
    # 2. Remove markdown code fences
    cleaned = cleaned.replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    
    # 3. Regex: find first { ... } block
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    
    # 4. All parsing failed
    return None

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
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")

    model = genai.GenerativeModel(GEMINI_MODEL)
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
    Use Gemini Vision to read and grade a handwritten answer in one shot.
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

    try:
        response = model.generate_content([
            {
                "mime_type": "image/jpeg",
                "data": image_b64
            },
            grading_prompt
        ])
        raw_text = response.text
        print(f"[Gemini Vision Grade] Raw response: {raw_text[:500]}")
        
        result = extract_json_from_text(raw_text)
        if result and "score" in result:
            return result
        else:
            print(f"[Gemini Vision Grade] Failed to extract JSON from: {raw_text[:500]}")
            return {"student_text": "", "score": 0, "feedback": "Error parsing Gemini response", "raw": raw_text}
    except Exception as e:
        print(f"[Gemini Vision Grade] API Error: {str(e)}")
        return {"student_text": "", "score": 0, "feedback": f"Gemini API error: {str(e)}"}


def grade_text_with_gemini(student_answer: str, expected_answer: str, question_context: Optional[str] = None) -> dict:
    """
    Use Gemini to semantically grade a typed text answer (no image needed).
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key not configured")

    model = genai.GenerativeModel(GEMINI_MODEL)

    grading_prompt = f"""You are an expert exam grader. Grade the student's answer against the expected answer.

Question: {question_context if question_context else "Determine from the answer context"}
Expected Answer: {expected_answer}
Student's Answer: {student_answer}

Instructions:
1. Compare the student's answer with the expected answer.
2. Rate on a scale of 0 to 100 based on accuracy and conceptual understanding.
3. Be fair - award marks for partially correct answers.
4. If the student's answer is essentially the same as the expected answer (even with minor differences in wording), give a score of 90-100.
5. Provide constructive feedback.

Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):
{{"score": number, "feedback": "your feedback"}}"""

    try:
        response = model.generate_content(grading_prompt)
        raw_text = response.text
        print(f"[Gemini Grade Text] Raw response: {raw_text[:500]}")
        
        result = extract_json_from_text(raw_text)
        if result and "score" in result:
            return result
        else:
            print(f"[Gemini Grade Text] Failed to extract JSON from: {raw_text[:500]}")
            return {"score": 0, "feedback": "Error parsing Gemini response", "raw": raw_text}
    except Exception as e:
        print(f"[Gemini Grade Text] API Error: {str(e)}")
        return {"score": 0, "feedback": f"Gemini API error: {str(e)}"}


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
        grading_result = grade_with_gemini_vision(contents, expected_answer, gemini_prompt)

        return {
            "student_text": grading_result.get("student_text", ""),
            "grading_result": grading_result
        }

    else:
        raise HTTPException(status_code=400, detail="Invalid grading method. Use 'similarity' or 'gemini'.")


@app.post("/grade-text")
async def grade_text(
    student_answer: str = Form(...),
    expected_answer: str = Form(...),
    method: str = Form("gemini"),  # "similarity" or "gemini"
    question_context: Optional[str] = Form(None)
):
    """
    Grade a typed text answer (no image needed).
    Used for short-answer exams where students type their response.
    """
    if method == "similarity":
        score_percentage = fuzz.token_sort_ratio(student_answer.lower(), expected_answer.lower())        
        final_marks = 100 if score_percentage >= 85 else 0

        return {
            "student_answer": student_answer,
            "expected_answer": expected_answer,
            "similarity_score": score_percentage,
            "final_marks": final_marks,
            "passed": score_percentage >= 85
        }

    elif method == "gemini":
        grading_result = grade_text_with_gemini(student_answer, expected_answer, question_context)       

        return {
            "student_answer": student_answer,
            "grading_result": grading_result
        }

    else:
        raise HTTPException(status_code=400, detail="Invalid grading method. Use 'similarity' or 'gemini'.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

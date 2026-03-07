import os
import io
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
import pdfplumber
from fuzzywuzzy import fuzz
from dotenv import load_dotenv

# ML Imports
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
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

# Initialize OCR Models (lazy-loaded for efficiency if needed)
# Using PaddleOCR for detection only (det=True, rec=False)
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', det=True, rec=False)

# TrOCR Model for Handwritten Recognition
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Configure Gemini
GENAMI_API_KEY = os.getenv("GEMINI_API_KEY")
if GENAMI_API_KEY:
    genai.configure(api_key=GENAMI_API_KEY)

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
    
    # Simple PDF parsing for Q&A (Assumes a structured format for now)
    # A real implementation would need more robust parsing/regex
    extracted_data = []
    with pdfplumber.open(file.file) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
        
        # Mock logic: split by "Question X:"
        # This is a place where you'd customize based on your PDF format
        extracted_data = full_text.split("Question") # Very naive split
        
    answer_keys = load_answer_keys()
    answer_keys[exam_id] = extracted_data
    save_answer_keys(answer_keys)
    
    return {"message": "Answer key uploaded and parsed", "exam_id": exam_id}

def perform_ocr(image: Image.Image):
    # Convert PIL to CV2 format for Paddle
    img_array = np.array(image.convert('RGB'))
    # PaddleOCR expects BGR for some internal ops if det is used
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 1. Detection with PaddleOCR
    result = paddle_ocr.ocr(img_cv2, cls=True)
    
    # 2. Recognition with TrOCR
    # result[0] contains list of [bbox, text_conf]
    detected_text = []
    
    # Extract bounding boxes and crop regions for TrOCR
    # Paddle returns: [[ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ], (text, score)]
    # Since we set rec=False, it's just [[ [x1,y1], ... ]]
    for line in result[0]:
        bbox = line[0]
        # Crop the region from the original image
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        left, top, right, bottom = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
        
        # Buffer crop slightly
        crop = image.crop((left-5, top-5, right+5, bottom+5))
        
        # Run TrOCR
        pixel_values = processor(images=crop, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        detected_text.append(text)
    
    return " ".join(detected_text)

@app.post("/grade-image")
async def grade_image(
    file: UploadFile = File(...),
    exam_id: str = Form(...),
    question_idx: int = Form(...),
    method: str = Form("similarity"), # "similarity" or "gemini"
    gemini_prompt: Optional[str] = Form(None)
):
    # 1. Read Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # 2. Perform OCR
    student_text = perform_ocr(image)
    
    # 3. Grading Logic
    answer_keys = load_answer_keys()
    if exam_id not in answer_keys:
        raise HTTPException(status_code=404, detail="Answer key for this exam not found")
    
    try:
        expected_answer = answer_keys[exam_id][question_idx]
    except IndexError:
        raise HTTPException(status_code=400, detail="Question index out of bounds")

    if method == "similarity":
        score_percentage = fuzz.token_sort_ratio(student_text.lower(), expected_answer.lower())
        final_marks = 100 if score_percentage >= 85 else 0 # Adjust logic as needed
        return {
            "student_text": student_text,
            "expected_answer": expected_answer,
            "similarity_score": score_percentage,
            "final_marks": final_marks,
            "passed": score_percentage >= 85
        }
    
    elif method == "gemini":
        if not GENAMI_API_KEY:
             raise HTTPException(status_code=500, detail="Gemini API Key not configured")
        
        # Simple grading prompt
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Grading Task:
        Question: {gemini_prompt if gemini_prompt else "Identify from context"}
        Expected Answer: {expected_answer}
        Student's Answer (OCR Result): {student_text}
        
        Rate the student's answer on a scale of 0 to 100 based on accuracy and conceptual understanding.
        Provide the result in JSON format: {{"score": number, "feedback": "string"}}
        """
        
        response = model_gemini.generate_content(prompt)
        try:
            grading_result = json.loads(response.text.replace('```json', '').replace('```', '').strip())
        except:
            grading_result = {"score": 0, "feedback": "Error parsing Gemini response", "raw": response.text}
            
        return {
            "student_text": student_text,
            "grading_result": grading_result
        }

    else:
        raise HTTPException(status_code=400, detail="Invalid grading method")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

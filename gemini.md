# EdulinkX Autograder - Production Architecture & Implementation Details

This document serves as the technical reference for the Autograder microservice, designed for deployment on AWS (EC2/App Runner) or a high-performance VPS.

## 1. System Architecture (The "95% Accuracy" Pipeline)
To handle messy handwriting and varied layouts, the backend uses a multi-stage hybrid pipeline:

```text
Image (Student Submission)
   ↓
[Stage 1: Detection] PaddleOCR 
   (Efficiently identifies text bounding boxes/regions)
   ↓
[Stage 2: Recognition] TrOCR (Vision Transformer)
   (Decodes handwritten text from cropped regions using 'trocr-base-handwritten')
   ↓
[Stage 3: Grading Strategy]
   ├─ A: Fuzzy Similarity (Threshold: 85-90%) - Best for Answer Keys
   └─ B: LLM Semantic Grading (Gemini 1.5 Flash) - Best for conceptual answers
```

## 2. Technical Stack
- **Backend:** FastAPI (Python 3.10+)
- **OCR Engine:** `paddleocr` (Detection) + `transformers` (TrOCR Recognition)
- **Math/Vision:** `opencv-python`, `Pillow`, `numpy`
- **Grading:** `fuzzywuzzy` (Levenshtein Distance), `google-generativeai`
- **PDF Processing:** `pdfplumber` (for Teacher Answer Keys)

## 3. Production Deployment Guide (AWS/VPS)

### Infrastructure Requirements
- **RAM:** Minimum 4GB (8GB recommended for TrOCR models).
- **CPU:** 2+ Cores.
- **GPU (Optional):** Highly recommended for faster inference, but CPU works for low-volume.

### Setup Steps
1. **Clone & Environment:**
   ```bash
   git clone https://github.com/jayant810/Autograder_Backend
   cd Autograder_Backend
   python -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Environment Variables (.env):**
   ```env
   GEMINI_API_KEY=your_api_key_here
   PORT=8000
   ```

3. **Running with Uvicorn:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## 4. API Endpoints Reference

### `POST /upload-answer-key`
- **Purpose:** Teacher uploads a PDF answer key for a specific exam.
- **Params:** `file` (PDF), `exam_id` (String).

### `POST /grade-image`
- **Purpose:** Grades a student's handwritten answer image.
- **Params:** 
  - `file`: Image (JPG/PNG)
  - `exam_id`: ID of the exam
  - `question_idx`: Index of the question in the answer key
  - `method`: `similarity` or `gemini`
  - `gemini_prompt`: (Optional) Specific context for Gemini grading.

## 5. Scaling to Production (HackerRank Style)
- **Sandboxing:** For future code-execution grading, use **Docker-in-Docker** with strict resource limits (`--memory=256m`, `--cpus=0.5`).
- **Async Queue:** Use **Redis + Celery** if you expect >10 simultaneous submissions to prevent API timeouts during OCR processing.
- **Security:** Implement an **API Gateway** or share the `JWT_SECRET` from the main EdulinkX backend to verify student identity before grading.

---
*Created on March 8, 2026, for EdulinkX Nexus.*

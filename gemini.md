# EdulinkX Autograder - Production Architecture & Implementation Details

This document serves as the technical reference for the Autograder microservice, designed for lightweight cloud deployment (Render, Railway, or any Docker host).

## 1. System Architecture (Gemini Vision Pipeline)
The backend uses Gemini's multimodal vision capabilities for both OCR and grading:

```text
Image (Student Submission)
   ↓
[Grading Strategy]
   ├─ A: Gemini Vision OCR → Fuzzy Similarity (Threshold: 85%)
   │      - Gemini extracts handwritten text from image
   │      - Text is compared to answer key using Levenshtein distance
   │      - Best for: Short, factual answers
   │
   └─ B: Gemini Vision Direct Grading (One-shot)
          - Gemini reads the image AND grades in a single API call
          - Evaluates accuracy + conceptual understanding
          - Best for: Long-form, conceptual answers
```

## 2. Technical Stack
- **Backend:** FastAPI (Python 3.10+)
- **OCR & Grading:** `google-generativeai` (Gemini 2.5 Flash-Lite)
- **Fuzzy Matching:** `fuzzywuzzy` (Levenshtein Distance)
- **PDF Processing:** `pdfplumber` (for Teacher Answer Keys)
- **Image Handling:** `Pillow`

## 3. Deployment Guide

### Infrastructure Requirements
- **RAM:** 512 MB (sufficient — no local ML models)
- **CPU:** 1 Core
- **Free Hosts:** Render, Railway, Koyeb

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
   ```

3. **Running Locally:**
   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. **Deploy to Render (Free):**
   - Push to GitHub
   - Go to [render.com](https://render.com) → New Web Service → Connect your repo
   - It auto-detects the `Dockerfile` and `render.yaml`
   - Add `GEMINI_API_KEY` in Environment settings
   - Deploy!

## 4. API Endpoints Reference

### `GET /`
- **Purpose:** Health check.
- **Response:** `{"status": "Autograder Microservice Online"}`

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

## 5. Rate Limits (Free Tier)
Using Gemini 2.5 Flash-Lite:
- **1,000 requests/day** (resets at midnight PT)
- **15 requests/minute**
- ~3 exams of 30 students × 10 questions per day

## 6. Future Scaling
- **Redis + Celery** for async grading queues
- **Upgrade to Gemini paid tier** for higher limits
- **Docker sandboxing** for code-execution grading (HackerRank-style)
- **API Gateway / JWT** for student identity verification

---
*Updated March 17, 2026 — Refactored to Gemini Vision pipeline for free cloud deployment.*

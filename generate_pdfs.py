import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import fpdf
except ImportError:
    install("fpdf")
    import fpdf

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

# 1. Generate Teacher Answer Key (Text PDF)
pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(200, 10, txt="Exam Answer Key", ln=1, align='C')
pdf.set_font('Arial', '', 12)

# Question 1: Short Answer
pdf.multi_cell(0, 10, txt="Question 1\nParis")
pdf.ln(5)

# Question 2: Long Answer
pdf.multi_cell(0, 10, txt="Question 2\nPhotosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water. It involves chlorophyll and generates oxygen as a byproduct.")

pdf.output("teacher_answer_key.pdf")
print("Generated teacher_answer_key.pdf")

# 2. Generate Student Submission (Image converted to PDF)
# We create an image with text to simulate a scanned handwritten document
img = Image.new('RGB', (800, 800), color=(255, 255, 255))
d = ImageDraw.Draw(img)

# Fallback to default font if we can't load a specific one
try:
    font = ImageFont.truetype("arial.ttf", 24)
except IOError:
    font = ImageFont.load_default()

d.text((50, 50), "Student Submission", fill=(0,0,0), font=font)

d.text((50, 120), "Question 1", fill=(0,0,0), font=font)
d.text((50, 160), "Paris", fill=(0,0,150), font=font)

d.text((50, 240), "Question 2", fill=(0,0,0), font=font)
# Manually wrapping text for the image
q2_ans = [
    "Photosynthesis is how plants make food.",
    "They use sunlight, carbon dioxide, and",
    "water to create nutrients. Oxygen is",
    "also produced during this process."
]
y_offset = 280
for line in q2_ans:
    d.text((50, y_offset), line, fill=(0,0,150), font=font)
    y_offset += 40

# Save the image as PDF
img.save("student_submission.pdf", "PDF", resolution=100.0)
print("Generated student_submission.pdf (Image-based PDF)")

import fitz # PyMuPDF
import sys

# Set standard output encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

pdf_path = r"c:\Users\dutta\Desktop\pdf-chatbot\backend\data\uploads\Aritra-Dutta-Resume.pdf"
doc = fitz.open(pdf_path)
print(f"Total Pages: {len(doc)}")
for i, page in enumerate(doc):
    print(f"--- Page {i+1} ---")
    print(page.get_text())

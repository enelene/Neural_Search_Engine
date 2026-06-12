import fitz 
import re

def extract_clean_text_from_pdf(pdf_path, skip_start_pages=25, skip_end_pages=80):
    print(f"Reading PDF from: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = []

    end_page = len(doc) - skip_end_pages

    for page_num in range(skip_start_pages, end_page):
        page = doc[page_num]
        rect = page.rect
        crop_box = fitz.Rect(
            rect.x0, 
            rect.y0 + (rect.height * 0.08), 
            rect.x1, 
            rect.y1 - (rect.height * 0.08)
        )
        
        raw_text = page.get_text("text", clip=crop_box)
        text = str(raw_text) if raw_text else ""
        
        text = re.sub(r'\d+\s+CHAPTER\s+\d+\s+•.*', ' ', text)
        
        text = re.sub(r'-\n\s*', '', text)
        
        text = text.replace('\n', ' ')
        
        text = re.sub(r'(\b\d+\b\s*){3,}', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if text:
            full_text.append(text)

    final_text = " ".join(full_text)
    print(f"Extracted {len(full_text)} pages of clean text.")
    return final_text
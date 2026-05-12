import fitz 
import re

def extract_clean_text_from_pdf(pdf_path, skip_start_pages=25, skip_end_pages=80):
    """
    Extracts text, removes textbook headers/footers, and cuts off the bibliography.
    """
    print(f"Reading PDF from: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = []

    # Calculate exactly where to stop (ignoring the index/bibliography)
    end_page = len(doc) - skip_end_pages

    for page_num in range(skip_start_pages, end_page):
        page = doc[page_num]
        rect = page.rect
        
        # Crop the top 8% and bottom 8% (removes page numbers and most headers)
        crop_box = fitz.Rect(
            rect.x0, 
            rect.y0 + (rect.height * 0.08), 
            rect.x1, 
            rect.y1 - (rect.height * 0.08)
        )
        
        raw_text = page.get_text("text", clip=crop_box)
        text = str(raw_text) if raw_text else ""
        
        # --- NEW AGGRESSIVE CLEANING RULES ---
        # 1. Remove Jurafsky specific headers like "18 CHAPTER 2 • WORDS AND TOKENS"
        text = re.sub(r'\d+\s+CHAPTER\s+\d+\s+•.*', ' ', text)
        
        # 2. Fix words broken across lines (e.g., "distrib-\nuted" -> "distributed")
        text = re.sub(r'-\n\s*', '', text)
        
        # 3. Replace newlines with spaces
        text = text.replace('\n', ' ')
        
        # 4. Remove excessive numbers (cleans up raw table data/matrices)
        # Replaces sequences of 3 or more numbers with a single space
        text = re.sub(r'(\b\d+\b\s*){3,}', ' ', text)
        
        # 5. Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        if text:
            full_text.append(text)

    final_text = " ".join(full_text)
    print(f"✅ Extracted {len(full_text)} pages of clean text.")
    return final_text
from src.utils import extract_clean_text_from_pdf
from src.pipeline import DataPipeline 
import os

PDF_PATH = "data/jurafsky_martin.pdf"
OUTPUT_DIR = "data/processed"

def main():
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: Please place the textbook PDF at {PDF_PATH}")
        return

    # 1. Extract and Clean
    print("Starting extraction...")
    raw_text = extract_clean_text_from_pdf(PDF_PATH)

    # 2. Chunk it up
    print("Starting chunking...")
    pipeline = DataPipeline(output_dir=OUTPUT_DIR)
    
    # Using an overlap of 20 words so context isn't lost between chunks
    chunks = pipeline.preprocess_and_chunk(raw_text, chunk_size=250, overlap=20)

    # 3. Save as JSON
    pipeline.save_data(chunks, "jurafsky_chunks", save_as_csv=False)
    
    print(f"Pipeline complete! Generated {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
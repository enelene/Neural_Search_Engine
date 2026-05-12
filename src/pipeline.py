import json
import pandas as pd
import re
import os

class DataPipeline:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def preprocess_and_chunk(self, raw_text, chunk_size=250, overlap=20):
        # Clean whitespace
        text = re.sub(r'\s+', ' ', raw_text).strip()
        words = text.split()
        
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_content = " ".join(words[i : i + chunk_size])
            
            # Calculate what percentage of the chunk is actual alphabet letters
            letters = sum(c.isalpha() for c in chunk_content)
            total_chars = len(chunk_content)
            
            if total_chars > 0 and (letters / total_chars) > 0.60: # Must be at least 60% real words
                chunk_id = f"chunk_{len(chunks) + 1:04d}"
                chunks.append({
                    "id": chunk_id,
                    "content": chunk_content,
                    "word_count": len(chunk_content.split())
                })
        return chunks

    def save_data(self, data, filename, save_as_csv=True):
        """
        Saves the list of dictionaries as either CSV or JSON.
        """
        filepath = os.path.join(self.output_dir, filename)
        
        if save_as_csv:
            df = pd.DataFrame(data)
            full_path = f"{filepath}.csv"
            df.to_csv(full_path, index=False)
            print(f"Data saved to {full_path}")
        else:
            full_path = f"{filepath}.json"
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Data saved to {full_path}")

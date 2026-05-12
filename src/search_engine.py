# src/search_engine.py
import json
import string
from rank_bm25 import BM25Okapi

class BM25Baseline:
    def __init__(self, data_path):
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # Extract just the text content for the search index
        self.corpus = [item["content"] for item in self.data]
        self.ids = [item["id"] for item in self.data]
        
        print("Tokenizing corpus and building BM25 Index (this might take a few seconds)...")
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("✅ Index ready!")

    def _tokenize(self, text):
        """Simple tokenizer: lowercase and remove punctuation."""
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def search(self, query, top_k=3):
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get the indices of the highest scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "id": self.ids[idx],
                "score": round(scores[idx], 4),
                "content": self.corpus[idx]
            })
        return results

if __name__ == "__main__":
    # Point this to the JSON file you just created!
    engine = BM25Baseline("data/processed/jurafsky_chunks.json")
    
    print("\n--- Jurafsky & Martin Search Engine ---")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_query = input("Enter search query: ")
        if user_query.lower() == 'exit':
            break
            
        print("\nSearching...")
        results = engine.search(user_query, top_k=3)
        
        for i, res in enumerate(results, 1):
            print(f"\n[{i}] Chunk ID: {res['id']} (Score: {res['score']})")
            # Print the first 300 characters to keep the terminal clean
            print(f"Text: {res['content'][:300]}...")
        print("-" * 50)
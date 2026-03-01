import json
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

# DISCLAIMER: Load the exact same model used for embedding
MODEL_NAME = "asafaya/albert-base-arabic"
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_query_embedding(text):
    """Embeds the user's search query using ALBERT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    
    return cls_embedding.squeeze().numpy().astype('float32')

def build_faiss_index(jsonl_file):
    print("Loading embeddings from disk...")
    
    all_chunks = []
    all_embeddings = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            chunks = data.get('text_chunks', [])
            embeddings = data.get('chunk_embeddings', [])
            
            if len(chunks) == len(embeddings):
                all_chunks.extend(chunks)
                all_embeddings.extend(embeddings)

    print(f"Loaded {len(all_chunks)} total text chunks.")
    
    embeddings_matrix = np.array(all_embeddings).astype('float32')
    
    dimension = embeddings_matrix.shape[1] 
    
    index = faiss.IndexFlatL2(dimension)
    
    index.add(embeddings_matrix) # type: ignore
    print(f"Successfully built FAISS index with {index.ntotal} vectors.")
    
    return index, all_chunks

def search(query, index, chunks, top_k=3):
    print(f"\nSearching for: '{query}'...")
    
    query_vector = get_query_embedding(query)
    
    query_vector = np.expand_dims(query_vector, axis=0)
    
    distances, indices = index.search(query_vector, top_k)
    
    print("-" * 50)
    for i in range(top_k):
        idx = indices[0][i]
        dist = distances[0][i]
        
        # Save into a file
        with open('4_generation/output/search_results.txt', 'a', encoding='utf-8') as f:
            f.write(f"Query: {query}\n")
            f.write(f"Result {i+1} (Distance: {dist:.4f}):\n{chunks[idx]}\n")
            f.write("-" * 50 + "\n")

if __name__ == "__main__":
    input_file = '3_embedding/output/embedded_youm7_data.json'
    
    faiss_index, text_chunks = build_faiss_index(input_file)
    
    # Test
    test_query = "الحرب في إيران"
    search(test_query, faiss_index, text_chunks, top_k=3)
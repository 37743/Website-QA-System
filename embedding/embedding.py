import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_config import get_config

config = get_config()

MODEL_NAME = config['model']['embed_model']

print(f"Loading tokenizer and model for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=int(config['embedding']['max_length']), chunk_size=int(config['embedding']['chunk_size']))
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    
    return cls_embedding.squeeze().tolist()

def process_embeddings(input_json, output_json):
    print("Generating embeddings...")
    with open(input_json, 'r', encoding='utf-8') as infile, \
         open(output_json, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            article_data = json.loads(line)
            
            chunk_embeddings = []
            
            for chunk in article_data.get('text_chunks', []):
                if chunk.strip():
                    embedding = get_bert_embedding(chunk)
                    chunk_embeddings.append(embedding)
            
            article_data['chunk_embeddings'] = chunk_embeddings
            
            json.dump(article_data, outfile, ensure_ascii=False)
            outfile.write('\n')

if __name__ == "__main__":
    input_file = 'preprocessing/output/processed_data.json'
    output_file = 'embedding/output/embedded_data.json'
    
    process_embeddings(input_file, output_file)
    print("Embeddings saved.")
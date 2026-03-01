import json
import torch
from transformers import AutoTokenizer, AutoModel

# Options: "asafaya/bert-mini-arabic", "asafaya/albert-base-arabic", "Geotrend/distilbert-base-ar-cased"
MODEL_NAME = "asafaya/albert-base-arabic"

print(f"Loading tokenizer and model for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
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
    input_file = '2_preprocessing/output/processed_youm7_data.json'
    output_file = '3_embedding/output/embedded_youm7_data.json'
    
    process_embeddings(input_file, output_file)
    print("Embeddings saved.")
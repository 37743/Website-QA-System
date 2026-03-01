import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Options: "asafaya/bert-mini-arabic", "asafaya/albert-base-arabic", "Geotrend/distilbert-base-ar-cased"
MODEL_NAME = "asafaya/albert-base-arabic"

print(f"Loading tokenizer and model for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def mean_pooling(model_output, attention_mask):
    """Averages token embeddings while ignoring [PAD] tokens."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    sentence_embedding = mean_pooling(outputs, inputs['attention_mask'])
    
    normalized_embedding = F.normalize(sentence_embedding, p=2, dim=1)
    
    return normalized_embedding.squeeze().tolist()

def process_embeddings(input_json, output_json):
    print("Generating embeddings...")
    with open(input_json, 'r', encoding='utf-8') as infile, \
         open(output_json, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
            
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
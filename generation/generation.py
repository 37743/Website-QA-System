import torch
import os
import sys
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, pipeline
from generation.search import build_faiss_index
from groq import Groq
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_config import get_config

config = get_config()

EMBED_MODEL = config['model']['embed_model']

# LLM_ID = config['model']['hf_model']
# with open('generation/hf_token.txt', 'r') as f:
#     HF_TOKEN = f.read().strip()

GROQ_MODEL = config['model']['groq_model']
with open('generation/g_token.txt', 'r') as f:
    G_TOKEN = f.read().strip()

# print("Loading Llama 3.1 8B...")
# model_kwargs = {
#     "torch_dtype": torch.bfloat16,
#     "quantization_config": {"load_in_4bit": True},
#     "device_map": "auto",
# }

# pipe = pipeline("text-generation", model=LLM_ID, model_kwargs=model_kwargs, token=HF_TOKEN)

client = Groq(api_key=G_TOKEN)

embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(EMBED_MODEL)

def get_query_embedding(text):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=int(config['embedding']['max_length']))
    with torch.no_grad():
        outputs = embed_model(**inputs)
    
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    
    return cls_embedding.squeeze().numpy().astype('float32')

# def run_rag(query, index, chunks):
#     vec = np.expand_dims(get_query_embedding(query), axis=0)
#     _, indices = index.search(vec, 3)
#     context = "\n\n".join([chunks[i] for i in indices[0]])

#     messages = [
#         {"role": "system", "content": "أنت مساعد خبير. استخدم المعلومات المقدمة فقط للإجابة."},
#         {"role": "user", "content": f"المعلومات:\n{context}\n\nالسؤال: {query}"}
#     ]

#     print("Generating answer...")
#     output = pipe(messages, max_new_tokens=256, temperature=0.2)
#     if isinstance(output, list) and len(output) > 0:
#         generated = output[0].get('generated_text', [])
#         if generated:
#             return generated[-1].get('content', "")
#     return ""

def run_rag_groq(query, index, chunks):
    vec = np.expand_dims(get_query_embedding(query), axis=0)

    scores, indices = index.search(vec, 10) 
    context = "\n\n".join([chunks[i] for i in indices[0]])

    print("Generating answer with Groq...")
    system_msg = "أنت مساعد خبير في الشؤون الإخبارية. استخدم المعلومات المقدمة فقط للإجابة."
    user_msg = f"المعلومات الإخبارية:\n{context}\n\nالسؤال: {query}"

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.2,
        max_tokens=512
    )

    return completion.choices[0].message.content

if __name__ == "__main__":
    input_file = 'embedding/output/embedded_data.json'
    faiss_index, text_chunks = build_faiss_index(input_file)
    # answer = run_rag("ما هي أخبار الحرب في إيران؟", faiss_index, text_chunks)
    answer = run_rag_groq("ما اخبار الحرب في إيران؟", faiss_index, text_chunks)
    with open('generation/output/answer.txt', 'w', encoding='utf-8') as f:
        f.write(answer if answer is not None else "")
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, pipeline
from search import build_faiss_index
from groq import Groq

EMBED_MODEL = "asafaya/albert-base-arabic"

# LLM_ID = "meta-llama/Llama-3.1-8B-Instruct"
# with open('4_generation/hf_token.txt', 'r') as f:
#     HF_TOKEN = f.read().strip()

GROQ_MODEL = "llama-3.1-8b-instant"
with open('4_generation/g_token.txt', 'r') as f:
    G_TOKEN = f.read().strip()

# print("Loading Llama 3.1 8B...")
# model_kwargs = {
#     "torch_dtype": torch.bfloat16,
#     "quantization_config": {"load_in_4bit": True},
#     "device_map": "auto",
# }

# pipe = pipeline("text-generation", model=LLM_ID, model_kwargs=model_kwargs, token=HF_TOKEN)

client = Groq(api_key=G_TOKEN)

print("Loading ALBERT model and tokenizer for embedding...")
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(EMBED_MODEL)

def get_query_embedding(text):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = embed_model(**inputs)
    mask = inputs['attention_mask'].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
    pooled = torch.sum(out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1).squeeze().numpy().astype('float32')

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
    scores, indices = index.search(vec, 3)
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
    input_file = '3_embedding/output/embedded_youm7_data.json'
    faiss_index, text_chunks = build_faiss_index(input_file)
    # answer = run_rag("ما هي أخبار الحرب في إيران؟", faiss_index, text_chunks)
    answer = run_rag_groq("ما أخبار الكرة المصرية؟", faiss_index, text_chunks)
    with open('4_generation/output/answer.txt', 'w', encoding='utf-8') as f:
        f.write(answer if answer is not None else "")
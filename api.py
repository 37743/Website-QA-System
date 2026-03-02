from fastapi.middleware.cors import CORSMiddleware

import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

from generation.search import build_faiss_index
from generation.generation import get_query_embedding, run_rag_groq
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_config import get_config

config = get_config()

EMBED_MODEL = config['model']['embed_model']
GROQ_MODEL = config['model']['groq_model']

print("Loading model and tokenizer for embedding...")
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
embed_model = AutoModel.from_pretrained(EMBED_MODEL)

app = FastAPI(title="Youm7 Arabic RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Building FAISS Index for API...")
input_file = 'embedding/output/embedded_data.json'
faiss_index, text_chunks = build_faiss_index(input_file)

class QueryRequest(BaseModel):
    query: str
    top_k: int = config.getint('search', 'top_k', fallback=3)

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]

@app.get("/")
def read_root():
    return {
        "status": "online", 
        "llm_model": GROQ_MODEL, 
        "embedding_model": EMBED_MODEL
    }

@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        vec = np.expand_dims(get_query_embedding(request.query), axis=0)
        _, indices = faiss_index.search(vec, request.top_k) # type: ignore
        
        retrieved_context = [text_chunks[i] for i in indices[0]]

        answer = run_rag_groq(request.query, faiss_index, text_chunks)

        return QueryResponse(
            query=request.query,
            answer=answer if answer else "لم يتم العثور على إجابة.",
            sources=retrieved_context
        )

    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
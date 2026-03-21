from fastapi.middleware.cors import CORSMiddleware

import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import subprocess
import json
import uvicorn
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Adjust paths to import your local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_config import get_config, save_config
from embedding.embedding import process_embeddings 
from generation.generation import run_rag_groq, retrieve_candidates, build_bm25_index

config = get_config()
GROQ_MODEL = config['model']['groq_model']

bm25 = None
metadata = None

app = FastAPI(title="E-JUST Sport AI RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]

@app.get("/")
def read_root():
    return {
        "status": "online",
        "llm_model": GROQ_MODEL,
        "retrieval": "BM25"
    }

@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    global bm25, metadata
    
    if bm25 is None or metadata is None:
        raise HTTPException(
            status_code=503, 
            detail="The API is currently building or updating the search index. Please try again in a moment."
        )

    try:
        results = retrieve_candidates(request.query, bm25, metadata, top_k=10)
        answer = run_rag_groq(request.query, bm25, metadata)

        sources = list({
            r.get("url") for r in results if r.get("url")
        })

        return QueryResponse(
            query=request.query,
            answer=answer if answer else "لم يتم العثور على إجابة.",
            sources=sources
        )

    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def merge_scrap_data(main_file, temp_file):
    if not os.path.exists(temp_file):
        return

    main_data = []
    if os.path.exists(main_file):
        with open(main_file, 'r', encoding='utf-8') as f:
            try:
                main_data = json.load(f)
            except json.JSONDecodeError:
                pass

    with open(temp_file, 'r', encoding='utf-8') as f:
        try:
            new_data = json.load(f)
        except json.JSONDecodeError:
            new_data = []

    if new_data:
        main_data.extend(new_data)
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(main_data, f, ensure_ascii=False, indent=4)
            
    os.remove(temp_file)


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    spider_dir = os.path.join(base_dir, 'data_collection', 'datacollection1') 
    input_file = os.path.join(spider_dir, 'output', 'scrap.json')
    temp_file = os.path.join(spider_dir, 'output', 'temp_scrap.json')
    output_file = os.path.join(base_dir, 'embedding', 'output', 'embedded_data.json')

    last_updated_str = config.get('system', 'last_updated', fallback='2026-03-19')
    system_time_str = config.get('system', 'system_time', fallback='2026-03-21')

    last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d")
    system_time = datetime.strptime(system_time_str, "%Y-%m-%d")
    
    target_end_date = system_time - timedelta(days=1)
    target_end_date_str = target_end_date.strftime("%Y-%m-%d")
    print(f"System time: {system_time_str}, Last updated: {last_updated_str}, Target end date for scraping: {target_end_date_str}")
    
    if last_updated <= target_end_date:
        print(f"Updates required! Scraping data from {last_updated_str} to {target_end_date_str}...")
        
        if not os.path.exists(spider_dir):
            print(f"ERROR: The spider directory doesn't exist at: {spider_dir}")
            sys.exit(1)

        try:
            subprocess.run([
                "scrapy", "crawl", "yallakorascrap",
                "-a", f"start_date={last_updated_str}",
                "-a", f"end_date={target_end_date_str}",
                "-O", temp_file
            ], cwd=spider_dir, check=True)
            
            print("Scraping complete. Merging data...")
            merge_scrap_data(input_file, temp_file)
            
            config['system']['last_updated'] = system_time.strftime("%Y-%m-%d")
            save_config(config)
            print(f"Config updated: 'last_updated' is now {target_end_date_str}")
            
            print("Running embeddings on updated data...")
            process_embeddings(input_file, output_file)
            print("Embeddings updated successfully.")
            
        except subprocess.CalledProcessError as e:
            print(f"Scraping failed: {e}")
            
    else:
        print(f"Data is up to date (last scraped up to {last_updated_str}). Skipping scraping and embedding.")

    print("Building BM25 Index for API...")
    if os.path.exists(output_file):
        bm25, metadata = build_bm25_index(output_file)
        print("Index built successfully.")
    else:
        print(f"Warning: Embedding file not found at {output_file}. Index is empty.")

    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
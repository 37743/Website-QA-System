import json
import os
import sys
import re
from rank_bm25 import BM25Okapi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_config import get_config

config = get_config()

print("Loading BM25 search...")

def normalize_ar(text):
    if not text:
        return ""
    text = text.lower()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text):
    return normalize_ar(text).split()

def build_bm25_index(jsonl_file):
    print("Loading match data...")

    texts = []
    metadata = []
    tokenized_corpus = []

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            text = data.get("text", "")
            tokens = tokenize(text)

            texts.append(text)
            tokenized_corpus.append(tokens)

            metadata.append({
                "team_a": data.get("team_a"),
                "team_b": data.get("team_b"),
                "score_a": data.get("score_a"),
                "score_b": data.get("score_b"),
                "date": data.get("date"),
                "time": data.get("time"),
                "url": data.get("url")
            })

    bm25 = BM25Okapi(tokenized_corpus)

    print(f"Loaded {len(texts)} matches into BM25.")

    return bm25, texts, metadata

def search(query, bm25, texts, metadata, top_k=5):
    print(f"\n🔍 Query: {query}")

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    output_path = os.path.join("generation", "output")
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, "search_results.txt")

    results = []

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\nQUERY: {query}\n")
        f.write("=" * 60 + "\n")

        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            score = scores[idx]

            result = {
                "rank": rank,
                "score": float(score),
                "team_a": metadata[idx]["team_a"],
                "team_b": metadata[idx]["team_b"],
                "score_a": metadata[idx]["score_a"],
                "score_b": metadata[idx]["score_b"],
                "date": metadata[idx]["date"],
                "time": metadata[idx]["time"],
                "details": texts[idx],
                "url": metadata[idx]["url"]
            }
            results.append(result)

            result_text = f"""
Result {rank} (Score: {score:.4f})
Teams: {metadata[idx]['team_a']} vs {metadata[idx]['team_b']}
Score: {metadata[idx]['score_a']}-{metadata[idx]['score_b']}
Date: {metadata[idx]['date']}
Time: {metadata[idx]['time']}
Details: {texts[idx]}
URL: {metadata[idx]['url']}
------------------------------------------------------------
"""

            print(result_text)
            f.write(result_text)

    return results

def build_context(results, max_chars=8000):
    context_parts = []

    for r in results:
        block = f"""مباراة رقم {r['rank']}:
الفريقان: {r['team_a']} ضد {r['team_b']}
النتيجة: {r['score_a']}-{r['score_b']}
التاريخ: {r['date']}
الوقت: {r['time']}
التفاصيل الكاملة: {r['details']}
الرابط: {r['url']}
-------------------------
"""
        context_parts.append(block)

    context = "\n".join(context_parts)

    if len(context) > max_chars:
        context = context[:max_chars]

    return context

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    input_file = os.path.join(base_dir, 'embedding', 'output', 'embedded_data.json')

    bm25, texts, metadata = build_bm25_index(input_file)

    results = search("مباراة الأهلي", bm25, texts, metadata, top_k=5)

    context = build_context(results)

    print("\nGenerated Context:\n")
    print(context)
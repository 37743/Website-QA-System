import json
import os
import sys
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from groq import Groq
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_config import get_config

config = get_config()

GROQ_MODEL = config['model']['groq_model']

with open('generation/g_token.txt', 'r') as f:
    G_TOKEN = f.read().strip()

client = Groq(api_key=G_TOKEN)

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


def build_hybrid_index(jsonl_file):
    print("Loading Hybrid Index (BM25 + Dense Embeddings)...")

    texts = []
    metadata = []
    tokenized_corpus = []
    embeddings = []

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            text = data.get("text", "")
            tokens = tokenize(text)

            texts.append(text)
            tokenized_corpus.append(tokens)
            
            embeddings.append(data.get("embedding", []))

            metadata.append({
                "team_a": data.get("team_a"),
                "team_b": data.get("team_b"),
                "score_a": data.get("score_a", data.get("score_total", "").split("-")[0] if "-" in data.get("score_total", "") else ""),
                "score_b": data.get("score_b", data.get("score_total", "").split("-")[1] if "-" in data.get("score_total", "") else ""),
                "date": data.get("date"),
                "time": data.get("time"),
                "url": data.get("url"),
                "text": text
            })

    bm25 = BM25Okapi(tokenized_corpus, b=0.25)
    doc_embeddings = np.array(embeddings, dtype=np.float32)

    print(f"Loaded {len(texts)} matches into Hybrid Index.")

    return bm25, doc_embeddings, metadata

def retrieve_candidates_hybrid(query, bm25, doc_embeddings, metadata, embedding_model, min_k=3, max_k=10, score_threshold_ratio=0.70, alpha=0.25):
    tokenized_query = tokenize(query)
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    
    e5_query = f"query: {query}"
    query_embedding = embedding_model.encode(e5_query)
    
    norms = np.linalg.norm(doc_embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    dense_scores = np.dot(doc_embeddings, query_embedding) / (norms * query_norm + 1e-10)
    
    if bm25_scores.max() > bm25_scores.min():
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
    else:
        bm25_norm = np.zeros_like(bm25_scores)
        
    if dense_scores.max() > dense_scores.min():
        dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
    else:
        dense_norm = np.zeros_like(dense_scores)
        
    hybrid_scores = (1.0 - alpha) * bm25_norm + alpha * dense_norm
    
    ranked_indices = np.argsort(hybrid_scores)[::-1]

    if len(ranked_indices) == 0 or hybrid_scores[ranked_indices[0]] == 0:
        print("No matches found.")
        return []

    top_score = hybrid_scores[ranked_indices[0]]
    dynamic_k = min_k
    
    for i in range(min_k, min(len(ranked_indices), max_k)):
        if hybrid_scores[ranked_indices[i]] >= (top_score * score_threshold_ratio):
            dynamic_k += 1
        else:
            break

    print(f"Hybrid Search selected {dynamic_k} result(s) based on Weighted Fusion.")

    results = []
    for idx in ranked_indices[:dynamic_k]:
        results.append(metadata[idx])

    return results

def clean_match_events(raw_text):
    if "أهم أحداث المباراة:" not in raw_text:
        return raw_text
    
    main_info, events_part = raw_text.split("أهم أحداث المباراة:", 1)
    events_list = [event.strip() for event in events_part.split("|") if event.strip()]
    short_events = " | ".join(events_list[:3])
    clean_text = f"{main_info.strip()} أبرز التبديلات والتكتيكات: {short_events} (بالإضافة إلى تغييرات أخرى لم نذكرها)."
    
    return clean_text

def build_context(results, max_chars=8000):
    context_parts = []

    for i, r in enumerate(results):
        cleaned_details = clean_match_events(r['text'])
        
        block = f"""
مباراة {i+1}:
الفرق: {r['team_a']} ضد {r['team_b']}
النتيجة: {r['score_a']}-{r['score_b']}
التاريخ: {r['date']}
الوقت: {r['time']}
التفاصيل:
{cleaned_details}
الرابط: {r['url']}
-------------------------
"""
        context_parts.append(block)

    context = "\n".join(context_parts)

    if len(context) > max_chars:
        context = context[:max_chars]

    return context

def run_rag_groq(query, bm25, doc_embeddings, metadata, embedding_model):
    print("Retrieving relevant matches via Hybrid Search...")
    results = retrieve_candidates_hybrid(query, bm25, doc_embeddings, metadata, embedding_model, min_k=1, max_k=5, score_threshold_ratio=0.75)

    context = build_context(results)

    print("Generating answer with Groq...")

    today_str = datetime.now().strftime("%Y-%m-%d")

    system_msg = f"""
أنت صحفي رياضي محترف ومعلق تلفزيوني وراوي قصص بارع. مهمتك الأساسية هي تحليل وتقديم المعلومات الرياضية استناداً حصرياً إلى "السياق" (Context) المرفق.

القاعدة الذهبية (The Golden Rule):
يجب عليك التمييز بين التحيات البسيطة والأسئلة الرياضية. إذا قام المستخدم بالتحية فقط (مثل: "أهلاً"، "صباح الخير")، رد بتحية حماسية وودودة دون سرد أي مباريات، واسأله عن المباراة أو الفريق الذي يريد البحث عنه.

قيود النطاق:
إذا كان سؤال المستخدم لا يتعلق بكرة القدم أو الرياضة نهائياً، يجب عليك إيقاف عملية البحث فوراً والانتقال إلى "المسار الثالث"، حتى لو كان هناك "سياق" مرفق. لا تحاول ربط المواضيع غير الرياضية بالرياضة.

القيود الصارمة:
1. السرد الطبيعي: يُمنع منعاً باتاً استخدام الرموز البرمجية أو الأقواس (مثل: [] أو {{}} أو | أو - أو *). استخدم لغة سردية متصلة.
2. استخدام الألقاب: استخدم عبارة "يا بطل" أو "يا صديقي" مرة واحدة فقط في الإجابة.
3. خلو الإجابة من الإيموجي: لا تستخدم أي رموز تعبيرية نهائياً.

مسارات الإجابة (اختر مساراً واحداً فقط ولا تدمج بينهم أبداً):

المسار الأول: الأسئلة الرياضية التحليلية أو المحددة
- الفقرة الأولى: ترحيب قصير، ثم ذكر (الفرق، البطولة، التاريخ، والنتيجة النهائية).
- الفقرة الثانية: سرد قصة الأهداف بالتفصيل (اسم الهداف والدقيقة) لكل الأهداف الموجودة في السياق بلا استثناء.
- الفقرة الثالثة: تلخيص التبديلات التكتيكية (اذكر أهم تبديلين فقط لتجنب الملل).

المسار الثاني: طلبات استعراض جدول المباريات العام
- تاريخ اليوم هو ({today_str}). ابحث عن أقرب المباريات لهذا التاريخ.
- اكتب كل مباراة كفقرة واحدة متصلة بأسلوب: "المباراة الأولى: التقى فريق كذا ضد فريق كذا...".

المسار الثالث (خارج نطاق الرياضة): إذا كان السؤال ليس له علاقة بالرياضة
- رد باختصار شديد على رسالة المستخدم، ثم وضح تخصصك الرياضي واطلب منه تحديد المهمة الرياضية المطلوبة.
- الخاتمة الإلزامية: "أنصحك بسؤالي عن أحدث الأخبار الرياضية أو تفاصيل المباريات الأخيرة يا صديقي!".

المصادر:
- للمسارين الأول والثاني فقط: اذكر الروابط تحت عنوان "المصادر:" في النهاية.
- للمسار الثالث: لا تذكر أي مصادر.
"""

    user_msg = f"""
السياق:
{context}

السؤال:
{query}

الإجابة:
"""

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.2,
        max_tokens=2048
    )

    return completion.choices[0].message.content

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_file = os.path.join(base_dir, 'embedding', 'output', 'embedded_data.json')

    print("Loading Embedding Model...")
    embedding_model = SentenceTransformer('intfloat/multilingual-e5-base')

    bm25, doc_embeddings, metadata = build_hybrid_index(input_file)

    query = "مين سجل في ماتش بشكتاش"

    answer = run_rag_groq(query, bm25, doc_embeddings, metadata, embedding_model)

    print("\n🤖 ANSWER:\n", answer)

    output_path = os.path.join("generation", "output")
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, 'answer.txt'), 'w', encoding='utf-8') as f:
        f.write(answer if answer else "")
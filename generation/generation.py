import json
import os
import sys
from rank_bm25 import BM25Okapi
from groq import Groq
from datetime import datetime
import re

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


def build_bm25_index(jsonl_file):
    print("Loading BM25 index...")

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
                "url": data.get("url"),
                "text": text
            })

    bm25 = BM25Okapi(tokenized_corpus)

    print(f"Loaded {len(texts)} matches into BM25.")

    return bm25, metadata

def retrieve_candidates(query, bm25, metadata, top_k=15):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []

    for idx in ranked_indices[:top_k]:
        results.append(metadata[idx])

    return results

def build_context(results, max_chars=8000):
    context_parts = []

    for i, r in enumerate(results):
        block = f"""
مباراة {i+1}:
الفرق: {r['team_a']} ضد {r['team_b']}
النتيجة: {r['score_a']}-{r['score_b']}
التاريخ: {r['date']}
الوقت: {r['time']}
التفاصيل:
{r['text']}
الرابط: {r['url']}
-------------------------
"""
        context_parts.append(block)

    context = "\n".join(context_parts)

    # 🔥 protect token limit
    if len(context) > max_chars:
        context = context[:max_chars]

    return context


# -----------------------------
# RAG WITH GROQ (UPDATED)
# -----------------------------
def run_rag_groq(query, bm25, metadata):
    print("Retrieving relevant matches...")
    results = retrieve_candidates(query, bm25, metadata, top_k=10)

    context = build_context(results)

    print("Generating answer with Groq...")


    # Get current date in a readable format
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Inject the date into the system message
    system_msg = f"""
أنت روبوت محادثة رياضي مبهج وحماسي، تعمل كمعلق تلفزيوني ومحلل بيانات في نفس الوقت. التزم حصرياً بالمعلومات الواردة في "السياق" المرفق.

قواعد النبرة والبيانات:
1. النبرة: حماسية ومبهجة، تحدث مع المستخدم مباشرة (مثل "يا بطل"، "يا صديقي")، وبدون استخدام إيموجي نهائياً.
2. توفر البيانات: لا تقل "لا توجد معلومات" إلا إذا كان السياق فارغاً تماماً.
3. البطولة: استخرج اسم الدوري من الرابط وترجمه للعربية.
4. شمولية التفاصيل: عند سرد المباريات، اذكر (الفرق، البطولة، التاريخ، الوقت، النتيجة، كل الأهداف، وكل الأحداث الهامة) بلا استثناء.

تحليل نوع الطلب (مسار الإجابة):
- المسار الأول (الأسئلة التحليلية والمحددة): إذا سأل المستخدم سؤالاً محدداً (مثل: "من سجل في مباراة كذا؟"، "ما تفاصيل مباراة فريق كذا؟"، أو "كم مرة تواجه الفريقان؟")، أجب بأسلوب سردي طبيعي ومباشر كأنك محلل رياضي يرد على سؤاله بناءً على "السياق" فقط، وتجاهل تنسيق القائمة الإلزامي.
- المسار الثاني (طلبات استعراض المباريات): إذا كان الطلب عاماً (مثل: "مباريات اليوم"، "ما هي أحدث المباريات؟")، استخدم **قواعد التنسيق الصارمة** الخاصة بالمعالجة البرمجية.

قواعد البحث (للمسار الثاني):
5. تاريخ اليوم هو ({today_str}). ابحث عن المباراة الأقرب لهذا التاريخ إن لم يحدد المستخدم.
6. إذا لم تتطابق أي مباراة، اقترح (3-5) مباريات من السياق.

قواعد التنسيق الصارمة (لأغراض المعالجة البرمجية Parsing - للمسار الثاني فقط):
- ممنوع كتابة أي مقدمات قبل القائمة. ابدأ الإجابة مباشرة بعلامة النجمة (*).
- يجب أن تُكتب كل مباراة كنقطة واحدة مستقلة تبدأ بعلامة (*).
- يمنع تماماً استخدام فواصل الأسطر (Line breaks / Enter) أو القوائم المتداخلة داخل النقطة الخاصة بالمباراة. يجب أن تكون كل نقطة عبارة عن سطر/فقرة متصلة.
- النموذج الإلزامي لكل نقطة مباراة:
* تفاصيل المواجهة: التقى [الفريق أ] ضد [الفريق ب] ضمن منافسات [اسم البطولة] بتاريخ [التاريخ] الساعة [الوقت]، وانتهت المواجهة بنتيجة [النتيجة]، سجل الأهداف: [اللاعب] لصالح [الفريق] في الدقيقة [الدقيقة] (تُكرر لكل هدف)، وأبرز الأحداث: [الحدث] للاعب [اللاعب] في الدقيقة [الدقيقة] (تُكرر لكل حدث).

الرسالة الختامية (إلزامي لكل المسارات):
- اكتب رسالة ترحيبية/ختامية حماسية واحدة فقط (مثل: "أتمنى لك مشاهدة ممتعة يا بطل!" أو "هذه كانت تغطيتنا يا صديقي!") في سطر نصي عادي مستقل (بدون علامة النجمة *) بعد انتهاء إجابتك أو قائمتك، وقبل المصادر مباشرة.

المصادر:
- اجمع الروابط في النهاية تحت عنوان "المصادر:" بحيث يكون كل رابط في سطر منفصل.
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

    bm25, metadata = build_bm25_index(input_file)

    query = "مين سجل في ماتش بشكتاش"

    answer = run_rag_groq(query, bm25, metadata)

    print("\n🤖 ANSWER:\n", answer)

    output_path = os.path.join("generation", "output")
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, 'answer.txt'), 'w', encoding='utf-8') as f:
        f.write(answer if answer else "")
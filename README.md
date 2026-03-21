[![LATEST](https://img.shields.io/badge/Based_on_Sports_Data-990000?style=for-the-badge&label=RAG&labelColor=FFFFFF&logoColor=EF4444)]()

</div>

# ⚽ Sports RAG Assistant

## Project Description

This project is an intelligent **Sports Question Answering System** that processes football-related data (matches, teams, leagues) using a **Retrieval-Augmented Generation (RAG)** pipeline.

The system retrieves relevant match data and generates accurate, context-aware answers in Arabic through a multi-stage pipeline:

1. Data Retrieval (matches, schedules, stats)
2. Semantic Processing
3. Answer Generation using LLM

It is deployed using a **FastAPI backend** and a visually interactive **Streamlit UI**.

---

## 🚀 Features

- **Sports-Aware QA System**
  - Ask about matches, teams, results, and schedules in Arabic

- **Semantic Retrieval**
  - Retrieves the most relevant match data using vector-based similarity

- **Context-Aware Answers**
  - Uses LLM (Llama 3.1) to generate natural responses

- **Arabic NLP Support**
  - Fully optimized for Arabic queries and football terminology

- **Fast Inference**
  - Efficient retrieval + generation pipeline for real-time answers

- **Interactive UI**
  - Custom football-themed Streamlit interface (RTL + styled chat)

---

## 🧱 RAG System Architecture

User Query (Arabic)
        ↓
FastAPI Backend
        ↓
Retriever (Top-K Matches)
        ↓
Context Builder
        ↓
LLM (Llama 3.1)
        ↓
Generated Answer + Sources

---

## 🖥️ Usage

### 1. Run Backend
python api.py

### 2. Launch UI
streamlit run ui.py

Then ask:

- "ماذا كانت نتيجة مباراة نادي الأهلي؟"
- "نتيجة مباراة برشلونة امبارح كانت ايه؟"
- "مين متصدر الدوري الإنجليزي؟"

---

## 📌 API Example

### Endpoint
POST /query

### Request
{
  "query": "متى مباراة الزمالك القادمة؟",
  "top_k": 15
}

### Response
{
  "answer": "...",
  "sources": ["url1", "url2"]
}

---

## 🧠 Models Used

- paraphrase-multilingual-MiniLM-L12-v2
- Llama-3.1-8B

---

## Credits

### Project Contributors
- @37743 - Yousef Gomaa

[^1]: Developed for E-JUST Natural Language Processing Course - 2026
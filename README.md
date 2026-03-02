<div align="center">
  
[![LATEST](https://img.shields.io/badge/Based_on_Youm7_News_Portal-990000?style=for-the-badge&label=RAG&labelColor=FFFFFF&logoColor=990000)](https://www.youm7.com/)

</div>

# Project Description

This system processes large-scale Arabic news data through a multi-stage pipeline: preprocessing, semantic embedding via sentence transformers, and lightning-fast inference. [^1] It is served through a modern FastAPI backend and a user-friendly interface.

## Features

- **Semantic Search**: Uses paraphrase-multilingual-MiniLM-L12-v2 for deep Arabic contextual understanding.
- **Lightning Inference**: Utilizes (Llama-3.1-8b) for quick response times.
- **Modular Architecture**: Decoupled preprocessing, embedding, and generation modules.

## Usage

### Initialize Backend
```bash
python main.py
```

### Launch UI
```bash
streamlit run ui.py
```

Then enter any Arabic news query. The system will retrieve the top relevant chunks and synthesize an answer.

## FAQ
- **How is Arabic text handled?**  
  We utilize the asafaya/albert-base-arabic (word-level) or multilingual-MiniLM (sentence-level) models specifically fine-tuned for Arabic script and semantics.

- **Is the search purely keyword-based?**  
  No, it is primarily semantic (vector-based), meaning it understands synonyms and context even if exact words don't match.

## Credits

### Project Contributors
- @37743 - Yousef Gomaa

[^1]: Developed for E-JUST Natural Language Processing Course - 2026

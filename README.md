# 🤖 B.A.I.T.: BITS AI Tutor  
*A multi-course RAG-based learning assistant for BITS Pilani.*

BAIT helps students prepare smarter by combining **topic-wise PYQ sorting**, **slide-based retrieval**, and an **AI tutor powered by Gemini 2.0 Flash + FAISS**.

---

## 🚀 Features

### 🐬 Topic-Wise PYQ Sorting
Automatically classifies PYQs into correct topics using MiniLM sentence embeddings.

### 🐬 FAISS-Powered Slide Search
All course slides are embedded and indexed in FAISS for ultra-fast semantic search.

### 🐬 RAG-Based Question Answering
Ask any question → BAIT retrieves relevant slide chunks → Gemini answers strictly using provided context.

### 🐬 PYQ Recommendations
For every query, BAIT returns the **Top 3 most similar PYQs**.

### 🐬 Clean Streamlit UI
- **Left:** Topics and sorted PYQs
- **Right:** Chatbot, RAG answers, PYQ suggestions, topic distribution pie chart
---
## 📁 Folder Structure
```
Project-BAIT/
│── app.py
│── requirements.txt
└── courses/
├── Professional Ethics/
│ ├── question_bank.json
│ ├── topics.json
│ └── faiss_store/
│ ├── index.faiss
│ └── index.pkl
├── Organizational Psychology/
└── Human Resource Development/
```
---

## 🧠 How BAIT Works

### Embeddings  
Uses `sentence-transformers/all-MiniLM-L6-v2` (CPU-friendly) to embed:
- Topics  
- PYQs  
- Student queries  

### 2️Toic Classification  
Each PYQ is assigned to the closest topic using cosine similarity.

### RAG Pipeline  
When a student asks a question:

1. Embed query  
2. FAISS retrieves top-k slide chunks  
3. Prompt built with **only** these chunks  
4. Gemini answers factually from context  

### PYQ Similarity Search  
Top 3 most similar previously-asked PYQs are shown.




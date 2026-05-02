
# 🚀 SaaS Support RAG System (Production-Ready Demo Project)

## 🧠 Overview

This project is a **production-style Retrieval-Augmented Generation (RAG) system** built to simulate a real-world SaaS customer support assistant.

The system allows users to ask natural language questions (e.g., *“What is the refund policy?”*) and generates accurate answers using:

* 📄 Internal knowledge base (PDF/TXT)
* 🔍 Semantic + keyword retrieval
* 🧠 LLM-based reasoning
* ⚙️ Multi-stage query processing pipeline

---

## 🎯 Problem Statement

Traditional chatbots:

* Hallucinate ❌
* Lack context ❌
* Cannot retrieve company-specific policies ❌

This system solves that by:

```text
User Query → Retrieve relevant knowledge → Generate grounded answer
```

---

## 🏗️ System Architecture

```text
User Query
   ↓
Query Rewriting (LLM)
   ↓
Multi-Query Generation
   ↓
Hybrid Retrieval
   ├── Vector Search (FAISS)
   └── Keyword Search (TF-IDF)
   ↓
Deduplication
   ↓
Metadata Filtering
   ↓
Reranking (Embedding-based scoring)
   ↓
Context Building
   ↓
LLM (Groq - Llama 3)
   ↓
Final Answer
```

---

## 🔥 Key Features

### ✅ 1. Query Rewriting

Transforms vague queries into structured queries.

```text
"What about refunds?"
→ "What is the refund policy?"
```

---

### ✅ 2. Multi-Query Retrieval

Generates multiple variations of a query to improve recall.

```text
"refund policy"
→ ["How do I get a refund?", "refund conditions", "money back rules"]
```

---

### ✅ 3. Hybrid Search (Production Pattern)

| Type             | Purpose                |
| ---------------- | ---------------------- |
| FAISS (vector)   | semantic understanding |
| TF-IDF (keyword) | exact match            |

👉 Combines **precision + recall**

---

### ✅ 4. Metadata Filtering

Each chunk is tagged:

```json
{"category": "refund"}
```

Then filtered at query time:

```text
refund query → only refund chunks
```

👉 Reduces noise significantly

---

### ✅ 5. Reranking Layer

After retrieval:

* Documents are scored again using embeddings
* Top relevant chunks are selected

```text
Retrieval → broad
Reranking → precise
```

---

### ✅ 6. Hallucination Control

Strict prompt rules:

```text
- Answer ONLY from context
- If not found → "I don't know"
```

👉 Prevents fake answers

---

### ✅ 7. Evaluation Pipeline

Custom evaluation using keyword matching:

```python
Accuracy: 3/3
```

👉 Simulates real-world testing

---

### ✅ 8. Persistent Vector Database

* FAISS index saved to disk
* Documents serialized via pickle

```text
Cold Start → Build index
Warm Start → Load index instantly
```

---

## 🛠️ Tech Stack

| Component      | Tool                   |
| -------------- | ---------------------- |
| LLM            | Groq (Llama 3)         |
| Embeddings     | SentenceTransformers   |
| Vector DB      | FAISS                  |
| Keyword Search | Scikit-learn (TF-IDF)  |
| Orchestration  | Custom Python pipeline |
| Data           | PDF + TXT              |

---

## 📂 Project Structure

```text
rag_project/
│
├── rag/
│   ├── loader.py
│   ├── chunker.py
│   ├── embedding.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── keyword_retriever.py
│   ├── query_rewriter.py
│   ├── multi_query_generator.py
│   ├── reranker.py
│   ├── rag_pipeline.py
│   ├── indexing.py
│
├── eval/
│   └── evaluate.py
│
├── utils/
│   └── metadata.py
│
├── data/
│   └── saas_doc.pdf
│
├── artifacts/
│   ├── faiss.index
│   └── documents.pkl
│
├── test.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Add API Key

Create `.env`:

```text
GROQ_API_KEY=your_api_key
```

---

### 4️⃣ Run the system

```bash
python -m rag.test
```

---

## 🧪 Evaluation

Run:

```bash
python -m eval.evaluate
```

Example output:

```text
Accuracy: 3/3
```

---

## 📌 Example Queries

```text
• What is the refund policy?
• Can I get refund after 20 days?
• What are pricing plans?
• What happens if payment fails?
```

---

## 🧠 Key Learnings

This project helped build deep understanding of:

* RAG system design
* Retrieval vs reranking
* Chunking strategies
* Hybrid search architecture
* LLM prompt control
* Debugging AI pipelines
* Production trade-offs (latency vs accuracy)

---

## ⚠️ Challenges Faced

* Metadata not persisted → incorrect filtering
* Mixing Document and string types
* Over-filtering → empty retrieval
* Reranker threshold tuning

👉 Solved using systematic debugging

---

## 🚀 Future Improvements

* Streaming responses
* FastAPI deployment
* Caching embeddings
* Better reranking (cross-encoder)
* UI (Streamlit)

---

## 💡 Final Note

This is not just a chatbot.

It is a **modular, demo production-oriented AI system** that demonstrates:

```text
System design > model usage
```

---


# Swiggy Annual Report RAG Question Answering System

A Retrieval-Augmented Generation (RAG) based AI system that answers user questions strictly using the Swiggy Annual Report (FY 2023–24).

This application ensures:
- No hallucination
- Context-grounded answers
- Page-level citation
- Strict document-based responses

---

## Objective

Build an AI system that can:

- Accept natural language questions
- Retrieve relevant information from the Swiggy Annual Report
- Generate accurate, context-grounded answers
- Refuse out-of-scope queries

The system must answer strictly from the document content.

---

## Data Source

**Document:** Swiggy Annual Report FY 2023–24  
**Source Link:**  
https://www.swiggy.com/corporate/wp-content/uploads/2024/10/Annual-Report-FY-2023-24-1.pdf


---

## 🏗️ System Architecture


PDF Extraction
↓
Intelligent Chunking
↓
Embedding Generation (Sentence Transformers)
↓
FAISS Vector Store
↓
Semantic Retrieval (Cosine Similarity)
↓
Gemini LLM (Strict Prompt)
↓
Final Grounded Answer


---

## 🔧 Tech Stack

- Python 3.12
- PyMuPDF (PDF Extraction)
- tiktoken (Token-aware chunking)
- Sentence Transformers (Embeddings)
- FAISS (Vector Database)
- Gemini API (LLM)
- Streamlit (GUI)
- dotenv (Environment Management)

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Shubham-Narvekar/Question_Answering_RAG_System_Assignment.git
cd Assignment
```
Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

Install Dependencies
```bash
pip install pymupdf
pip install tiktoken
pip install sentence-transformers
pip install faiss-cpu
pip install numpy
pip install tqdm
pip install google-genai
pip install python-dotenv
```

Environment Variables

Create a .env file in project root:

```
GEMINI_API_KEY=your_api_key_here
```

⚠️ Do NOT push .env to GitHub.

🚀 Running the Application
Run Streamlit GUI
```bash
streamlit run app.py
```

The app will open at:

http://localhost:8501

🧪 Example Questions

What was Swiggy's consolidated net loss after tax for FY24?

How many cities does Swiggy offer Food Delivery services in?

How many Board meetings were held during FY24?

How many active dark stores does Swiggy's Instamart operate?

Out-of-scope test:

What is Swiggy’s market valuation?

🛡️ Hallucination Prevention Strategy

Cosine similarity threshold filtering

Top-k semantic retrieval

Strict grounding prompt

Temperature set to 0

Explicit refusal instruction

Context-only answer generation

📂 Project Structure
Assignment/
│
├── app.py
│
├── processing/
│   ├── extract_pdf.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   
│
├── data/
├── output/
├── .env
└── requirements.txt

📌 Features

Semantic search with FAISS

Metadata preservation (page numbers)

Overlapping token-aware chunking

Context display in UI

Grounded financial QA

Out-of-scope rejection

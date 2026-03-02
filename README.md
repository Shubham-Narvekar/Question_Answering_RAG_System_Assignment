# Swiggy Annual Report RAG System

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** based Question Answering system that allows users to ask natural language questions strictly based on the Swiggy Annual Report (FY 2023–24).

The system retrieves relevant sections from the document using semantic search and generates grounded answers using Google's Gemini LLM.

The model is strictly constrained to answer only from the document context and does not hallucinate.

---

## Dataset

- **Document**: Swiggy Annual Report FY 2023–24
- **Format**: PDF
- **Source Link**:  
  https://www.swiggy.com/corporate/wp-content/uploads/2024/10/Annual-Report-FY-2023-24-1.pdf


---

## System Architecture
PDF → Text Extraction → Smart Chunking → Embeddings → FAISS Vector Store → Query Embedding → Semantic Retrieval → Gemini LLM → Grounded Answer

---

## Key Features

- Section-aware intelligent chunking
- Token-based overlapping chunks
- Cosine similarity search using FAISS
- Semantic retrieval using sentence-transformers
- Strict grounding prompt
- Hallucination prevention
- Page number citation
- CLI + Streamlit GUI support

---

## Tech Stack

- Python
- PyMuPDF (PDF extraction)
- Sentence-Transformers (Embeddings)
- FAISS (Vector database)
- Google Gemini (LLM)
- Streamlit (GUI)
- dotenv (Environment management)

---
## Project Structure

Assignment/
│
├── app.py
│
├── processing/
│ ├── init.py
│ ├── extract_pdf.py
│ ├── chunking.py
│ ├── embeddings.py
│ ├── retriever.py
│ └── rag_pipeline.py
│
├── output/
│ ├── chunks.json
│ └── faiss_index/
│
├── data/
├── .env
├── requirements.txt
└── README.md

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Shubham-Narvekar/Question_Answering_RAG_System_Assignment.git
cd Assignment

### Install Dependencies

```bash
pip install pymupdf
pip install tiktoken
pip install sentence-transformers
pip install faiss-cpu
pip install numpy
pip install tqdm

### Add Gemini API Key
Create a .env file in project root:
```bash
GEMINI_API_KEY=your_api_key_here

### Run CLI version

```bash
python processing/rag_pipeline.py

### Run GUI version

```bash
streamlit run app.py


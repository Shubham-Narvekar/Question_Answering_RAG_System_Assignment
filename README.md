# Swiggy RAG Application

This project is a Retrieval-Augmented Generation (RAG) application designed to answer questions based on the content of a provided PDF document (e.g., Swiggy's Annual Report). It leverages FAISS for vector search and Sentence Transformers for semantic embeddings.

## Features
- Extracts and chunks text from PDF documents
- Generates embeddings for each chunk using a transformer model
- Builds a FAISS index for efficient similarity search
- Retrieves relevant document sections in response to user queries
- CLI interface for interactive Q&A
- Each result includes a similarity score, page number, section, and a text preview
- **Source document:** [Annual Report FY 2023-24 (PDF)](https://www.swiggy.com/corporate/wp-content/uploads/2024/10/Annual-Report-FY-2023-24-1.pdf)

## Project Structure
```
app.py                       # Main entry point (if applicable)
data/                        # Source documents (PDFs)
  Annual-Report-FY-2023-24...pdf
output/                      # Output data (chunks, embeddings, index)
  chunks.json
  raw_extracted.json
  faiss_index/
    metadata.json
    swiggy_faiss.index
processing/                  # Core processing modules
  __init__.py
  chunking.py                # PDF chunking logic
  embeddings.py              # Embedding generation
  extract_pdf.py             # PDF extraction logic
  rag_pipeline.py            # End-to-end RAG pipeline
  retriever.py               # FAISS-based retriever (CLI interface)
```

## How It Works
1. **PDF Extraction:** Extracts text from the source PDF.
2. **Chunking:** Splits the extracted text into manageable chunks.
3. **Embedding:** Generates vector embeddings for each chunk using Sentence Transformers (`all-mpnet-base-v2`).
4. **Indexing:** Stores embeddings in a FAISS index for fast similarity search.
5. **Retrieval:** On user query, embeds the query and retrieves the most relevant chunks from the index.
6. **Display:** Shows top results with similarity score, page number, section, and a text preview.

## Usage
1. Place your PDF document in the `data/` directory.
2. Run the processing pipeline to extract, chunk, embed, and index the document (see `processing/` scripts).
3. Start the retriever CLI:
   ```sh
   python processing/retriever.py
   ```
4. Enter your question at the prompt. Type `exit` to quit.

## Requirements
- Python 3.8+
- [faiss-cpu](https://github.com/facebookresearch/faiss)
- [sentence-transformers](https://www.sbert.net/)
- numpy


## Source Document
- [Annual Report FY 2023-24 (PDF)](https://www.swiggy.com/corporate/wp-content/uploads/2024/10/Annual-Report-FY-2023-24-1.pdf)

## License
This project is for educational and demonstration purposes only.


# IntelliVault – Personal Knowledge Vault using RAG

## Project Overview

IntelliVault is a Personal Knowledge Vault that enables users to store and query their own documents using semantic search and intelligent recall. Instead of relying on keyword-based search, the system allows users to ask natural language questions and receive context-aware answers grounded strictly in their uploaded documents.

The system is built using a Retrieval-Augmented Generation (RAG) approach, where relevant document content is retrieved first and then used to generate accurate and meaningful answers.

---

## Problem Statement

Students and professionals store large amounts of information in digital documents such as PDFs and notes. Over time, recalling specific information becomes difficult due to inefficient keyword-based search and lack of contextual understanding.

This project addresses this problem by enabling semantic retrieval from personal documents, allowing users to query their knowledge base effectively.

---

## Solution Approach

The system follows a Retrieval-Augmented Generation (RAG) architecture:

* Users upload documents (PDF or TXT)
* Documents are processed and split into smaller overlapping chunks
* Each chunk is converted into vector embeddings using a transformer model
* Embeddings are stored in a FAISS vector database

When a user submits a query:

* The query is converted into an embedding
* The system retrieves the most relevant chunks using semantic similarity
* A context-based answer is generated from the retrieved content

This ensures answers are grounded in the uploaded documents, reducing hallucination and improving reliability.

---

## Key Features

* Support for PDF and TXT document upload
* Automatic text extraction and preprocessing
* Configurable chunking with overlap
* Embedding generation using Sentence Transformers
* Vector storage and similarity search using FAISS
* Semantic retrieval based on meaning rather than keywords
* Context-aware answer generation
* Source-based answer transparency
* REST API using FastAPI
* Interactive frontend using Streamlit

---

## System Architecture

User
→ Frontend (Streamlit)
→ Backend (FastAPI)
→ Document Processing (Chunking & Embeddings)
→ Vector Database (FAISS)
→ Semantic Retrieval
→ Answer Generation
→ Response with Sources

---

## Tech Stack

| Layer               | Technology                               |
| ------------------- | ---------------------------------------- |
| Language            | Python                                   |
| Backend             | FastAPI                                  |
| Frontend            | Streamlit                                |
| Embeddings          | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database     | FAISS                                    |
| Document Processing | PyMuPDF / PyPDF                          |
| LLM / QA Logic      | Local extractive QA + optional LLM       |
| Version Control     | Git & GitHub                             |

---

## Project Structure

```
IntelliVault/
├── backend/
│   ├── ingestion/        # Document loading, chunking, embeddings
│   ├── retrieval/        # FAISS and search logic
│   ├── llm/              # Answer generation logic
│   ├── api.py            # FastAPI endpoints
│   ├── rag_pipeline.py   # End-to-end RAG pipeline
│
├── frontend/
│   ├── app.py            # Streamlit UI
│
├── data/
│   ├── uploads/          # Uploaded files
│   ├── processed_chunks/ # Stored chunk data
│   ├── vector_db/        # FAISS index
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How It Works

1. A user uploads a document through the frontend
2. The backend extracts text and splits it into chunks with overlap
3. Each chunk is converted into embeddings using a transformer model
4. Embeddings are stored in a FAISS vector database

When a query is asked:

* The query is converted into an embedding
* The system retrieves the most relevant chunks using similarity search
* Relevant sentences are selected and combined to generate a concise answer

The system returns the answer along with the source chunks.

---

## Improvements Implemented

* Fixed embedding and query normalization for better retrieval accuracy
* Improved FAISS indexing and consistency between chunks and vectors
* Enhanced retrieval ranking to return more relevant context
* Refined answer generation to reduce repetition and improve clarity
* Added error handling and stability improvements in API
* Ensured consistent backend module imports

---

## Future Improvements

* Integration of advanced LLMs for generative answers
* Hybrid search combining keyword and semantic retrieval
* Multi-document querying and filtering
* Improved UI/UX and performance optimization

---

## Team Roles

Backend & AI Logic:
Document processing, embeddings, vector database, retrieval, RAG pipeline

Frontend & Documentation:
Streamlit UI, API integration, report preparation, demonstration

---

## Authors

Kavyashi Trar
Prakrit

---














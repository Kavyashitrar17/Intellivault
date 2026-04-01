# IntelliVault – Personal Knowledge Vault using RAG

## Project Overview

**IntelliVault** is an AI-powered Personal Knowledge Vault that enables users to store, search, and retrieve information from their own documents using **semantic search** and **Retrieval-Augmented Generation (RAG)**.

Unlike traditional keyword-based systems, IntelliVault understands the *context and meaning* of queries, allowing users to ask natural language questions and receive **accurate, context-aware answers grounded strictly in their data**.

---

## Problem Statement

Students and professionals accumulate large volumes of unstructured data (PDFs, notes, documents). Traditional search systems fail because:

* They rely on exact keyword matches
* They lack contextual understanding
* They cannot generate precise answers

Result: **Time-consuming manual searching and poor knowledge recall**

---

## Solution

IntelliVault solves this using a **RAG-based architecture**, which combines:

* Semantic Retrieval (FAISS + embeddings)
* Context-aware Answer Generation

This ensures:

* Accurate answers
* Reduced hallucination
* Answers strictly grounded in user documents

---

## System Architecture

```
User Query
   ↓
Streamlit Frontend
   ↓
FastAPI Backend
   ↓
RAG Pipeline
   ├── Document Processing (Chunking)
   ├── Embedding Generation
   ├── Vector Storage (FAISS)
   ├── Semantic Retrieval
   └── Answer Generation
   ↓
Response + Source Context
```

---

## How It Works

### Document Ingestion

* Upload PDF/TXT files
* Extract text using PyMuPDF / PyPDF
* Split into **overlapping chunks**
* Convert chunks → embeddings
* Store in **FAISS vector database**

### Query Processing

* Convert user query → embedding
* Perform **semantic similarity search**
* Retrieve top relevant chunks

### Answer Generation

* Extract relevant context
* Generate concise answer
* Return answer + source references

---

## Key Features

* PDF & TXT document support
* Intelligent chunking with overlap
* Semantic search (not keyword-based)
* Fast retrieval using FAISS
* Context-aware answer generation
* Source-backed responses (transparency)
* REST API with FastAPI
* Interactive UI using Streamlit

---

## Tech Stack

| Layer            | Technology                               |
| ---------------- | ---------------------------------------- |
| Language         | Python                                   |
| Backend          | FastAPI                                  |
| Frontend         | Streamlit                                |
| Embeddings       | llama-3.1-8b-instant                     |
| Vector Database  | FAISS                                    |
| Document Parsing | PyMuPDF / PyPDF                          |
| AI Pipeline      | RAG (Retrieval-Augmented Generation)     |
| Version Control  | Git & GitHub                             |

---

## Project Structure

```
IntelliVault/
├── backend/
│   ├── ingestion/        # Document processing & embeddings
│   ├── retrieval/        # FAISS search logic
│   ├── llm/              # Answer generation
│   ├── api.py            # FastAPI endpoints
│   ├── rag_pipeline.py   # Core pipeline
│
├── frontend/
│   ├── app.py            # Streamlit UI
│
├── data/
│   ├── uploads/          
│   ├── processed_chunks/ 
│   ├── vector_db/        
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Improvements & Optimizations

* Fixed embedding normalization issues → better accuracy
* Improved FAISS indexing consistency
* Enhanced retrieval ranking
* Reduced repetition in generated answers
* Added API error handling & stability improvements

---

## Future Enhancements

* Integration with advanced LLMs (GPT / open-source models)
* Hybrid search (keyword + semantic)
* Multi-document querying
* Real-time document updates
* UI/UX enhancements

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/intellivault.git
cd intellivault
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Backend

```bash
uvicorn backend.api:app --reload
```

### 4. Run Frontend

```bash
streamlit run frontend/app.py
```

---

## 📡 API Overview

| Endpoint  | Description      |
| --------- | ---------------- |
| `/upload` | Upload documents |
| `/query`  | Ask questions    |
| `/health` | Check API status |

---

## Use Cases

* Student notes search
* Research paper analysis
* Personal knowledge management
* Document Q&A system

---

## Team

**Backend & AI:**

* RAG pipeline, embeddings, FAISS, API

**Frontend & Documentation:**

* UI development, integration, reporting

---

## Authors

* Kavyashi Trar
* Prakrit Mishra

---

## Key Highlight

> IntelliVault transforms static documents into an **intelligent, searchable knowledge system**, enabling users to interact with their data conversationally.

---

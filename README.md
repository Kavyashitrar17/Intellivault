# 🧠 IntelliVault – Personal Knowledge Vault using RAG

## 📌 Project Overview

IntelliVault is a **Personal Knowledge Vault** that enables users to store their own documents and retrieve information from them using **semantic search and intelligent recall**.
Instead of relying on keyword-based search, the system allows users to ask **natural language questions** and receive **context-aware answers grounded strictly in their uploaded documents**.

The project is built using a **Retrieval-Augmented Generation (RAG)** approach, where relevant document content is retrieved first and then passed to a Large Language Model (LLM) for reasoning.

---

## 🎯 Problem Statement

Students and professionals store large amounts of information in digital documents such as PDFs and notes. Over time, recalling specific information becomes difficult due to poor keyword-based search and lack of context.
This project aims to solve this problem by enabling **semantic recall from personal documents**, allowing users to query their own knowledge base intelligently.

---

## 💡 Solution Approach

The system follows a **Retrieval-Augmented Generation (RAG)** architecture:

1. User documents are uploaded and processed.
2. Documents are split into smaller chunks.
3. Each chunk is converted into vector embeddings.
4. Embeddings are stored in a vector database.
5. When a user asks a question:

   * Relevant chunks are retrieved using semantic similarity.
   * An LLM generates an answer **only using the retrieved content**.

This ensures **accurate, explainable, and non-hallucinated responses**.

---

## 🧩 Key Features (MVP Scope)

* Upload PDF or text documents
* Chunk and embed documents
* Store embeddings in a vector database
* Semantic search based on meaning, not keywords
* Context-grounded answer generation using LLM
* Display source document for answers

---

## 🏗 System Architecture (High Level)

```
User
 ↓
Frontend (Streamlit)
 ↓
Backend (FastAPI)
 ↓
Document Chunking & Embeddings
 ↓
Vector Database (FAISS)
 ↓
LLM (Answer Generation)
 ↓
Answer + Source
```

---

## 🛠 Tech Stack (Planned)

| Layer           | Technology                    |
| --------------- | ----------------------------- |
| Language        | Python                        |
| Backend         | FastAPI                       |
| Frontend        | Streamlit                     |
| Embeddings      | SentenceTransformers / OpenAI |
| Vector DB       | FAISS                         |
| LLM             | GPT / LLaMA (via API)         |
| Version Control | Git & GitHub                  |

---

## 📁 Project Structure

```
IntelliVault/
├── backend/        # AI & backend logic
├── frontend/       # Streamlit UI
├── data/           # Uploaded files & vector DB
├── docs/           # Diagrams & documentation
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📅 Week-1 Progress

* Finalized project idea and scope (MVP)
* Designed system architecture
* Created GitHub repository
* Implemented clean folder structure
* Added initial Streamlit frontend (UI skeleton)
* Enabled team collaboration via GitHub

---
## 📅 Week-2 Progress

*Implemented backend document ingestion module supporting PDF and TXT files
*Extracted clean raw text from uploaded documents
*Implemented text chunking logic with configurable chunk size and overlap
*Added metadata to each chunk (chunk ID and source document)
*Tested ingestion and chunking on long academic documents
*Verified correct generation of multiple chunks for large input
*Improved UI layout for clarity and usability
*Developed initial Streamlit frontend UI with:
      >File upload interface
      >Question input field
      >Placeholder sections for answers and sources


## 🚀 Future Work

* Implement document ingestion and chunking
* Generate and store embeddings
* Build semantic retrieval pipeline
* Integrate LLM for answer generation
* Improve UI and add error handling

---

## 👥 Team Roles

* **Backend & AI Logic:** Document processing, embeddings, vector search, RAG pipeline
* **Frontend & Documentation:** Streamlit UI, integration, report preparation, demo

---









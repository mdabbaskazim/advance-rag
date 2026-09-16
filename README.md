# Self-RAG v2

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A self-reflective retrieval-augmented generation (RAG) application built with Python, FastAPI, LangGraph, Google Gemini, and FAISS. It answers questions using internal company PDFs, evaluates whether the answer is supported by retrieved context, and rewrites the query when the initial answer is weak or not useful.

## Overview

This project is designed for business-document Q&A. It loads PDF files from the `documents/` folder, creates a vector index with FAISS, and uses a multi-step reasoning workflow to:

- decide whether retrieval is needed
- retrieve relevant document chunks
- evaluate relevance and support
- revise or rewrite the retrieval query if needed
- judge whether the final answer is useful
- expose the workflow through a FastAPI API

The backend uses Google Gemini models for embeddings and generation, and it stores the vector index locally in a `faiss_index/` directory to avoid re-indexing on every run.

## Features

- PDF ingestion and chunking with LangChain
- FAISS-based semantic retrieval
- LangGraph orchestration for retrieval and answer verification
- Self-check loop for answer support and usefulness
- Query rewriting for better retrieval quality
- FastAPI endpoint for chat-style requests
- Health check endpoint for service monitoring

## Tech Stack

- Python 3.11+
- FastAPI
- LangChain / LangGraph
- FAISS
- Google Generative AI
- Pydantic
- PyPDF
- python-dotenv

## Project Structure

```text
self-rag-v2/
├── .env
├── .gitignore
├── pyproject.toml
├── README.md
├── uv.lock
├── documents/
│   ├── Company_Policies.pdf
│   ├── Company_Profile.pdf
│   └── Product_and_Pricing.pdf
├── faiss_index/            # generated locally after first run
├── src/
│   └── self_rag_v2/
│       ├── __init__.py
│       ├── main.py
│       └── rag_engine.py
└── self_rag_.ipynb
```

## Prerequisites

Before running the project, make sure you have:

- Python 3.11 or newer
- A Google API key with access to Gemini models
- The project dependencies installed

## Installation

Using `uv` (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Environment Variables

Create a `.env` file in the project root with your Google API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

The app also accepts `GEMINI_API_KEY` as an alternative environment variable name.

## Running the Application

Start the FastAPI server:

```bash
uv run uvicorn self_rag_v2.main:app --reload
```

If you are running in a manually activated virtual environment:

```bash
uvicorn self_rag_v2.main:app --reload
```

The API will be available at:

- http://127.0.0.1:8000
- Swagger docs: http://127.0.0.1:8000/docs

## API Endpoints

### POST /api/chat

Send a question and receive an answer with supporting evidence.

Request body:

```json
{
  "question": "Describe NexaAI's company culture."
}
```

Example response:

```json
{
  "answer": "NexaAI emphasizes a collaborative and growth-focused culture ...",
  "evidence": [
    "...retrieved supporting text..."
  ],
  "is_useful": true,
  "rewrite_tries": 0
}
```

### GET /health

Returns the server health status:

```json
{
  "status": "healthy"
}
```

## How the Retrieval Pipeline Works

The application runs a LangGraph workflow that follows this flow:

1. Decide whether the question requires document retrieval.
2. Retrieve candidate documents from the FAISS index.
3. Filter which documents are relevant to the question.
4. Generate an answer from the relevant context.
5. Check whether the answer is truly supported by the retrieved evidence.
6. If the answer is weak or unsupported, revise or rewrite the retrieval query.
7. Evaluate whether the final answer is useful for the user's question.

This creates a more robust self-checking RAG pipeline than a simple one-shot retriever-generator setup.

## Document Setup

Place your PDF files inside the `documents/` folder before running the app. The project is currently configured to look for:

- `documents/Company_Policies.pdf`
- `documents/Company_Profile.pdf`
- `documents/Product_and_Pricing.pdf`

If the FAISS index has not been built yet, it will be created automatically the first time the app runs.

## Notes

- The vector index is cached in `faiss_index/`.
- The app uses `GOOGLE_API_KEY` or `GEMINI_API_KEY` from the environment.
- If the PDF files are missing, retrieval will not produce answers from the internal knowledge base.
- You can adjust the model, chunk size, retrieval count, and rewrite logic inside `src/self_rag_v2/rag_engine.py`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

Md Abbas Kazim

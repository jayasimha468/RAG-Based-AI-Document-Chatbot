# RAG-Based AI Document Chatbot

## Project Overview
A RAG application that allows users to upload PDF documents and ask questions based on their content.

## Features
- PDF upload and text extraction
- Text chunking with overlap
- Embeddings using Sentence Transformers
- Semantic search using FAISS
- Keyword search using BM25
- Hybrid retrieval using Reciprocal Rank Fusion
- Answer generation using Ollama Llama 3.2
- Retrieved context and source page display
- Chat history and clear history

## Technologies Used
- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- BM25
- NumPy
- Ollama

## Workflow
1. Upload a PDF document.
2. Extract text using PyMuPDF.
3. Split text into overlapping chunks.
4. Generate embeddings.
5. Retrieve information using FAISS and BM25.
6. Combine results using RRF.
7. Generate answers using Ollama.
8. Display answers and source pages.

## Installation
Install dependencies using: `pip install -r requirements.txt`

Download the Ollama model: `ollama pull llama3.2`

Run the application: `python -m streamlit run app.py`

## Testing
Tested with a 190-page PDF containing 578 chunks.

## Future Improvements
- Support multiple document formats.
- Improve retrieval relevance.
- Add evaluation metrics.
- Deploy the application online.

## Author
Jai
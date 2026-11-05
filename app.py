
import streamlit as st
import fitz
import faiss
import requests
import re
import unicodedata

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# Streamlit page configuration
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG-Based AI Document Chatbot")
st.write("Upload a PDF and ask questions based on its content.")


# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# Extract and split PDF text into chunks
def extract_chunks(pdf_bytes, chunk_size=1000, overlap=150):

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    chunks = []
    metadata = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text").strip()

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size
            chunk = text[start:end].strip()

            if len(chunk) > 50:
                chunks.append(chunk)
                metadata.append({"page": page_number})

            start += chunk_size - overlap

    document.close()

    return chunks, metadata


# Hybrid Search using FAISS + BM25 + RRF
def hybrid_search(query, model, index, bm25, chunks, k=5):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    faiss_scores, faiss_indices = index.search(
        query_embedding,
        min(10, len(chunks))
    )

    tokenized_query = re.findall(
        r'\b\w+\b',
        unicodedata.normalize("NFKC", query).lower()
    )

    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_indices = bm25_scores.argsort()[::-1][:10]

    # Reciprocal Rank Fusion
    rrf_scores = {}
    rrf_constant = 60

    for rank, idx in enumerate(
        faiss_indices[0],
        start=1
    ):
        rrf_scores[int(idx)] = (
            rrf_scores.get(int(idx), 0)
            + 1 / (rrf_constant + rank)
        )

    for rank, idx in enumerate(
        bm25_indices,
        start=1
    ):
        rrf_scores[int(idx)] = (
            rrf_scores.get(int(idx), 0)
            + 1 / (rrf_constant + rank)
        )

    ranked_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:k]

    return [idx for idx, score in ranked_results]


# PDF upload
uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


# Process uploaded PDF
if st.button("Process PDF"):

    if uploaded_file is None:

        st.warning("Please upload a PDF first.")

    else:

        with st.spinner("Processing PDF..."):

            try:

                pdf_bytes = uploaded_file.getvalue()

                chunks, metadata = extract_chunks(pdf_bytes)

                if not chunks:

                    st.error(
                        "No readable text found in this PDF."
                    )

                else:

                    # Normalize PDF text
                    normalized_chunks = [
                        unicodedata.normalize("NFKC", chunk)
                        for chunk in chunks
                    ]

                    # Tokenize chunks for BM25
                    tokenized_chunks = [
                        re.findall(r'\b\w+\b', chunk.lower())
                        for chunk in normalized_chunks
                    ]

                    # Create BM25 index
                    bm25 = BM25Okapi(tokenized_chunks)

                    # Generate embeddings
                    model = load_embedding_model()

                    embeddings = model.encode(
                        chunks,
                        normalize_embeddings=True
                    ).astype("float32")

                    # Create FAISS index
                    index = faiss.IndexFlatIP(
                        embeddings.shape[1]
                    )

                    index.add(embeddings)

                    # Store data in Streamlit session
                    st.session_state["chunks"] = chunks
                    st.session_state["metadata"] = metadata
                    st.session_state["index"] = index
                    st.session_state["model"] = model
                    st.session_state["bm25"] = bm25
                    st.session_state["file_name"] = uploaded_file.name

                    st.success("PDF processed successfully!")

                    st.write("Total chunks:", len(chunks))
                    st.write("Total pages:", len(set(
                        item["page"] for item in metadata
                    )))

            except Exception as e:

                st.error(
                    f"PDF processing failed: {e}"
                )


# Question-answering section
if "index" in st.session_state:

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if st.button("Clear Chat History"):
        st.session_state["chat_history"] = []
        st.rerun()

    st.divider()

    st.subheader("Ask a question")

    question = st.text_input(
        "Enter your question about the PDF:"
    )

    if st.button("Get Answer"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner(
                "Searching document and generating answer..."
            ):

                try:

                    # Load stored components
                    model = st.session_state["model"]
                    index = st.session_state["index"]
                    chunks = st.session_state["chunks"]
                    metadata = st.session_state["metadata"]
                    bm25 = st.session_state["bm25"]

                    # Perform Hybrid Search
                    k = min(5, len(chunks))

                    indices = hybrid_search(
                        question,
                        model,
                        index,
                        bm25,
                        chunks,
                        k
                    )

                    # Prepare retrieved context
                    context_parts = []
                    source_pages = set()

                    for idx in indices:

                        page_number = metadata[idx]["page"]

                        source_pages.add(page_number)

                        context_parts.append(
                            f"Source: Page {page_number}\n"
                            f"Content: {chunks[idx]}"
                        )
                    context = "\n\n".join(context_parts)

                    # Display retrieved document context
                    with st.expander("View Retrieved Context"):
                        for i, idx in enumerate(indices, start=1):
                            st.markdown(f"**Retrieved Chunk {i}**")
                            st.write("Page:", metadata[idx]["page"])
                            st.write(chunks[idx])
                            st.divider()

                    # Construct prompt
                    prompt = f"""
You are a document question-answering assistant.

Answer the question using only the provided document context.
Do not invent information.
If the answer is not available in the context, clearly say so.
Mention the source page numbers.

Document context:
{context}

Question:
{question}

Answer:
"""

                    # Generate answer using Ollama
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "llama3.2",
                            "prompt": prompt,
                            "stream": False
                        },
                        timeout=180
                    )

                    response.raise_for_status()

                    answer = response.json()["response"]

                    st.session_state["chat_history"].append({
                        "question": question,
                        "answer": answer,
                        "pages": ", ".join(
                            str(page) for page in sorted(source_pages)
                        )
                    })

                    st.subheader("AI Answer")

                    st.write(answer)

                    st.subheader("Source Pages")

                    st.write(
                        ", ".join(
                            str(page)
                            for page in sorted(source_pages)
                        )
                    )

                    st.divider()
                    st.subheader("Chat History")

                    for i, chat in enumerate(
                        st.session_state["chat_history"], start=1
                    ):
                        st.markdown(
                            f"**Question {i}:** {chat['question']}"
                        )
                        st.markdown(
                            f"**Answer:** {chat['answer']}"
                        )
                        st.caption(
                            f"Source Pages: {chat['pages']}"
                        )
                        st.divider()

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot connect to Ollama. "
                        "Make sure Ollama is running."
                    )

                except Exception as e:

                    st.error(
                        f"Error generating answer: {e}"
                    )

else:

    st.info(
        "Upload and process a PDF to start asking questions."
    )


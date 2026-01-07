import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="RPI",
    layout="centered"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Railway Info Bot")
st.sidebar.markdown("""
### Purpose
This assistant explains **railway passenger rules, station processes, and facilities**
using a **preloaded internal knowledge base**.

### Supported
- Ticket types (General, Tatkal, etc.)
- Boarding rules
- Platform usage
- Station facilities
- Passenger guidelines
""")

# ---------------- MAIN UI ----------------
st.title("Railway Passenger Information & Station Process Explainer Bot")
st.markdown(
    "Ask general railway-related questions. "
    "This system provides **explanations only**, not live or transactional services."
)

# ---------------- GEMINI CONFIG ----------------
genai.configure(api_key="AIzaSyCisz1kKfSejY8C1uTHxFdfJv2THZLhSB8")
chat_model = genai.GenerativeModel("models/gemini-2.5-flash")

# ---------------- LOCAL EMBEDDING MODEL ----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

DATA_FOLDER = "data"

# ---------------- LOAD DOCUMENTS ----------------
@st.cache_data(show_spinner=False)
def load_documents():
    docs = []
    for file in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, file)

        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())

        elif file.endswith(".pdf"):
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            docs.append(text)

    return docs

# ---------------- TEXT CHUNKING ----------------
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# ---------------- BUILD FAISS VECTOR DB ----------------
@st.cache_resource(show_spinner=False)
def build_vector_db():
    documents = load_documents()
    chunks = []

    for doc in documents:
        chunks.extend(chunk_text(doc))

    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    return index, chunks

index, all_chunks = build_vector_db()

# ---------------- RETRIEVAL ----------------
def retrieve_chunks(query, k=5):
    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True
    )

    _, indices = index.search(
        query_embedding.astype("float32"),
        k
    )

    return [all_chunks[i] for i in indices[0]]

# ---------------- CHAT HISTORY ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Ask a railway-related question")

if user_input:
    st.session_state.chat.append({"role": "user", "content": user_input})

    forbidden_keywords = [
        "live", "status", "platform now", "delay",
        "book", "booking", "cancel", "fare", "price"
    ]

    if any(word in user_input.lower() for word in forbidden_keywords):
        assistant_reply = (
            "This assistant does not support real-time information, "
            "ticket booking, cancellation, or fare calculation.\n\n"
            "Please use official railway services for these requests."
        )
    else:
        with st.spinner("Retrieving railway guidelines..."):
            context_chunks = retrieve_chunks(user_input)
            context_text = "\n\n".join(context_chunks)

            response = chat_model.generate_content(
                f"""
You are a Railway Passenger Information Assistant.

STRICT RULES:
- Answer ONLY using the railway knowledge base below.
- Explain passenger rules, station processes, and facilities only.
- DO NOT provide booking, ticket purchase, cancellation,
  fare calculation, or real-time train tracking.
- Use very simple English and bullet points.
- If the information is not present, say clearly that it is unavailable.

RAILWAY KNOWLEDGE BASE (Retrieved Context):
{context_text}

QUESTION:
{user_input}
"""
            )

            assistant_reply = response.text

    st.session_state.chat.append(
        {"role": "assistant", "content": assistant_reply}
    )
    st.rerun()

# ---------------- FOOTER ----------------
st.caption(
    "Informational use only. "
    "For bookings, fares, or live train status, please use official railway services."
)

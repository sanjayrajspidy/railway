import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os


st.set_page_config(
    page_title="RPI",
    layout="centered"
)


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

st.title("Railway Passenger Information & Station Process Explainer Bot")
st.markdown(
    "Ask general railway-related questions. "
    "This system provides **explanations only**, not live or transactional services."
)


genai.configure(api_key="AIzaSyAbdUHy_SpJiFKSWB_JSZGH_K5KpviEWrY")
model = genai.GenerativeModel("models/gemini-2.5-flash")


DATA_FOLDER = "data"
document_text = ""

for file in os.listdir(DATA_FOLDER):
    file_path = os.path.join(DATA_FOLDER, file)

    if file.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            document_text += f.read() + "\n"

    elif file.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            document_text += page.extract_text() or ""

if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    st.chat_message(msg["role"]).markdown(msg["content"])


user_input = st.chat_input(
    "Ask a railway-related question",
)


if user_input:
    # Store user message
    st.session_state.chat.append(
        {"role": "user", "content": user_input}
    )

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
        with st.spinner("Explaining using railway guidelines..."):
            response = model.generate_content(
                f"""
You are a Railway Passenger Information Assistant.

STRICT RULES:
- Answer ONLY using the railway knowledge base below.
- Explain passenger rules, station processes, and facilities only.
- DO NOT provide booking, ticket purchase, cancellation,
  fare calculation, or real-time train tracking.
- Use very simple English and bullet points.
- If the information is not present, say clearly that it is unavailable.

RAILWAY KNOWLEDGE BASE:
{document_text}

QUESTION:
{user_input}
"""
            )

        assistant_reply = response.text

    # Store assistant reply
    st.session_state.chat.append(
        {"role": "assistant", "content": assistant_reply}
    )

    st.rerun()


st.caption(
    "Informational use only. "
    "For bookings, fares, or live train status, please use official railway services."
)

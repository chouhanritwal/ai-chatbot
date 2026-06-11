import streamlit as st
import google.generativeai as genai
import fitz  # pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

st.title("DRDO AI Chatbot")

# Hindi toggle
hindi_mode = st.toggle("Reply in Hindi 🇮🇳")

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Reading PDF..."):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        st.session_state.vectorstore = vectorstore
        st.success("PDF loaded successfully!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.spinner("Thinking..."):

        # Language instruction
        if hindi_mode:
            language_instruction = "IMPORTANT: You must reply completely in Hindi language (Devanagari script). Even if the document is in English, your answer must be in Hindi only."
        else:
            language_instruction = "Reply in English."

        if "vectorstore" in st.session_state:
            docs = st.session_state.vectorstore.similarity_search(user_input, k=15)
            context = "\n".join([d.page_content for d in docs])
            prompt = f"""You are a helpful AI tutor. Use the document as a reference but explain answers in simple, clear language.
Do NOT just copy text from the document. Explain it like a teacher would to a student.
If the answer has multiple parts or a list, include ALL of them in the correct order.
Add examples where helpful. Be conversational.
{language_instruction}

Document context:
{context}

Question: {user_input}

Important: Give the COMPLETE answer with ALL points. Keep original numbering."""
        else:
            prompt = f"""{language_instruction}
You are a helpful AI assistant. Answer the following question clearly and completely.

Question: {user_input}"""

        response = model.generate_content(prompt)
        reply = response.text

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
# app/main_chatbot.py
import sys
import os
import streamlit as st

# Add the project root directory (one level up from /app)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.Loading_reports import process_and_store
from utils.summarize_text import extract_important_details
from app.rag_pipeline import get_rag_chain
from app.memory import get_memory

st.set_page_config(page_title="Healthcare Assistant Chatbot")
st.title("🤖 Healthcare Diagnostic Assistant")

# Get RAG and memory chains
rag_chain = get_rag_chain()
memory = get_memory()

# Upload patient file
uploaded_file = st.file_uploader("Upload patient report (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    # Ensure the 'temp/' folder exists
    os.makedirs("temp", exist_ok=True)

    # Save uploaded file
    file_path = f"temp/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract and embed
    text = process_and_store(file_path)

    # Display summary
    st.subheader("📄 Extracted Summary")
    summary = extract_important_details(text)
    st.info(summary)

# Chat interface
st.markdown("---")
st.subheader("💬 Ask questions from patient data")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input query
query = st.text_input("Ask a question:")

# Answer query
if query:
    # Retrieve documents related to the query (using retriever)
    documents = rag_chain.retriever.get_relevant_documents(query)
    
    # Use memory to combine the documents and query properly
    result = rag_chain.run({
        'query': query,  # Ensure the key is 'query'
        'input_documents': documents  # Pass the documents as expected
    })
    
    st.session_state.chat_history.append((query, result))
    st.success(result)

# Display conversation history
if st.session_state.chat_history:
    st.markdown("### 📝 Conversation History")
    for q, a in st.session_state.chat_history:
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {a}")

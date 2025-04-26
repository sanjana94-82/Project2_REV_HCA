import os
import streamlit as st
from utils.Loading_reports import process_and_store
from utils.summarize_text import extract_important_details
from app.components.rag_pipeline import get_rag_chain

def show():
    st.title("🧠 Chatbot Assistant for Healthcare Providers")

    uploaded_file = st.file_uploader("📄 Upload Patient Report (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

    # Only process if a new file is uploaded
    if uploaded_file:
        file_path = os.path.join("temp", uploaded_file.name)

        if st.session_state.get("uploaded_file_name") != uploaded_file.name:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            full_text = process_and_store(file_path)
            summary = extract_important_details(full_text)

            st.session_state.full_text = full_text
            st.session_state.summary = summary
            st.session_state.uploaded_file_name = uploaded_file.name

    # If a file has already been processed, show summary and chat interface
    if st.session_state.get("summary"):
        st.subheader("📌 Summary of Report")
        st.markdown(st.session_state.summary)

        st.markdown("---")
        query = st.text_input("Ask a medical question based on the report")

        if query:
            rag_chain = get_rag_chain(st.session_state.full_text)
            response = rag_chain.invoke({"query": query})

            st.markdown("### 🤖 Assistant's Response")
            if isinstance(response, dict) and "result" in response:
                st.markdown(response["result"])
            else:
                st.error("⚠️ Unexpected response format. Here's the raw response:")
                st.write(response)

    else:
        st.info("Please upload a patient report to begin.")
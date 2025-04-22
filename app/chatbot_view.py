import os
import streamlit as st
from utils.Loading_reports import process_and_store
from utils.summarize_text import extract_important_details
from app.rag_pipeline import get_rag_chain

def show():
    st.title("🧠 Chatbot Assistant for Healthcare Providers")

    uploaded_file = st.file_uploader("📄 Upload Patient Report (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file:
        file_path = os.path.join("temp", uploaded_file.name)

        # Only process if a new file is uploaded or if it's a different file than the previous one
        if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            # New file uploaded — reset and process
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Process and store the file content
            full_text = process_and_store(file_path)
            summary = extract_important_details(full_text)

            # Store in session state
            st.session_state.full_text = full_text
            st.session_state.summary = summary
            st.session_state.uploaded_file_name = uploaded_file.name

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
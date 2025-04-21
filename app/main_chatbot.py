import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.Loading_reports import process_and_store
from utils.summarize_text import extract_important_details
from app.rag_pipeline import get_rag_chain

st.set_page_config(page_title="Healthcare Diagnostic Assistant", layout="wide")
st.title("🩺 Healthcare Diagnostic Assistant for Providers")

uploaded_file = st.file_uploader("Upload patient report (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])

if "patient_summary" not in st.session_state:
    st.session_state.patient_summary = None
if "full_text" not in st.session_state:
    st.session_state.full_text = None

if uploaded_file:
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    text = process_and_store(file_path)
    summary = extract_important_details(text)
    st.session_state.full_text = text
    st.session_state.patient_summary = summary

    st.subheader("📄 Extracted Patient Summary")
    st.info(summary)

elif st.session_state.patient_summary:
    st.subheader("📄 Extracted Patient Summary")
    st.info(st.session_state.patient_summary)

st.markdown("---")
st.subheader("💬 Ask medical questions based on the uploaded report")

query = st.text_input("Enter your question:")

if query:
    if not st.session_state.full_text:
        st.warning("Please upload a report first.")
    else:
        rag_chain = get_rag_chain()
        try:
            result = rag_chain.invoke({"query": query})
            st.success(result["result"])
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
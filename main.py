import streamlit as st
from app import chatbot_view, differential_diagnosis

st.set_page_config(page_title="Healthcare Diagnostic Assistant", layout="wide")

# Ensure that session state is persistent across pages
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "full_text" not in st.session_state:
    st.session_state.full_text = None
if "summary" not in st.session_state:
    st.session_state.summary = None

st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Choose a module", ["Chatbot Assistant", "Differential Diagnosis"])

if page == "Chatbot Assistant":
    chatbot_view.show()
elif page == "Differential Diagnosis":
    differential_diagnosis.show()

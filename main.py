import streamlit as st

st.set_page_config(page_title="Healthcare Diagnostic Assistant", layout="wide")

from app import chatbot_view, differential_diagnosis, pubmed_screener

# Initialize session state variables
def init_session_state():
    defaults = {
        "uploaded_file_name": None,
        "full_text": None,
        "summary": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Create Tabs for Navigation
tab1, tab2, tab3 = st.tabs(["🧠 Chatbot Assistant", "🩺 Differential Diagnosis", "📚 PubMed Screener"])

try:
    with tab1:
        chatbot_view.show()
    with tab2:
        differential_diagnosis.show()
    with tab3:
        pubmed_screener.show()
except Exception as e:
    st.error(f"⚠️ An error occurred while loading the module: {e}")

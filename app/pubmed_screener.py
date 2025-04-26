# app/pubmed_screener.py

import sys
import os
import streamlit as st

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.components.chat_utils import ChatAgent
from app.components.prompts import chat_prompt_template
from app.components.llm import llm

def show():
    # Layout: logo on left, description on right
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image('assets/pubmed-screener-logo.jpg')

    with col2:
        st.title("PubMed Screener")
        st.markdown("""
            PubMed Screener is a ChatGPT & PubMed powered insight generator from biomedical abstracts. 

            #### Example scientific questions
            - How can advanced imaging techniques and biomarkers be leveraged for early diagnosis and monitoring of disease progression in neurodegenerative disorders?
            - What are the potential applications of stem cell technology and regenerative medicine in the treatment of neurodegenerative diseases, and what are the associated challenges?
            - What are the roles of gut microbiota and the gut-brain axis in the pathogenesis of type 1 and type 2 diabetes, and how can these interactions be modulated for therapeutic benefit?
            - What are the molecular mechanisms underlying the development of resistance to targeted cancer therapies, and how can these resistance mechanisms be overcome?
        """)

    # Instantiate chat agent
    chat_agent = ChatAgent(prompt=chat_prompt_template, llm=llm)

    # Display message history
    chat_agent.display_messages()

    # Get user input using Streamlit's native chat_input
    user_input = st.chat_input("Ask a biomedical question...")

    if user_input:
        # Show user's message
        st.chat_message("human").write(user_input)

        # Prepare config to pass session ID
        config = {"configurable": {"session_id": "session"}}

        # Invoke the chain safely
        with st.spinner("Searching PubMed..."):
            try:
                response = chat_agent.chain.invoke({"question": user_input}, config)
                if hasattr(response, "content"):
                    st.chat_message("ai").write(response.content)
                else:
                    st.chat_message("ai").write(str(response))
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

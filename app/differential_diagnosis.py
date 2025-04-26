#different_diagnosis.py
import os
import sys
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.components.rag_pipeline import get_rag_chain  # use same logic as the chatbot

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def show():
    st.title("🩻 Differential Diagnosis Assistant")

    if "full_text" not in st.session_state or not st.session_state.full_text:
        st.warning("Please upload a report in the Chatbot tab first.")
        return

    # Show diagnostic summary once
    if "diagnostic_summary" not in st.session_state:
        prompt_template = PromptTemplate.from_template("""
You're a top-tier diagnostic AI trained on complex medical cases. A healthcare provider has uploaded a detailed patient report. Based solely on this report and without assumptions, provide a ranked list of **differential diagnoses** with reasoning.

Apart from the diagnosis extracted from the patient report you will be able to provide other possible different diagnosis by understanding minute details of that report.

Report:
{report_text}

Output format:
1. **Primary Diagnosis**: [Diagnosis]  
   Reasoning: [Detailed clinical rationale]  

2. **Secondary Possibility**: [Diagnosis]  
   Reasoning: [...]

3. **Less Likely but Consider**: [Diagnosis]  
   Reasoning: [...]

Also mention if any urgent/life-threatening conditions only if it is needed and must be ruled out and why.
""")
        prompt = prompt_template.format(report_text=st.session_state.full_text)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.4,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        response = llm.invoke(prompt)
        st.session_state.diagnostic_summary = response.content

    st.markdown("### 🔬 Diagnostic Reasoning")
    st.markdown(st.session_state.diagnostic_summary)

    # Add a chatbot below
    st.markdown("---")
    st.subheader("💬 Ask about the Differential Diagnosis")

    query = st.text_input("Ask a question related to the diagnosis...")
    if query:
        rag_chain = get_rag_chain(st.session_state.full_text + "\n\n" + st.session_state.diagnostic_summary)
        response = rag_chain.invoke({"query": query})
        st.markdown("### 🤖 Assistant's Response")
        if isinstance(response, dict) and "result" in response:
            st.markdown(response["result"])
        else:
            st.write(response)
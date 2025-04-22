
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def show():
    st.title("🩻 Differential Diagnosis Assistant")

    # Check if full_text is in session state
    if "full_text" not in st.session_state or not st.session_state.full_text:
        st.warning("Please upload a report in the Chatbot tab first.")
        return

    detailed_prompt = PromptTemplate.from_template("""
You're a top-tier diagnostic AI trained on complex medical cases. A healthcare provider has uploaded a detailed patient report. Based solely on this report and without assumptions, provide a ranked list of **differential diagnoses** with reasoning.
Apart from the diagnosis mentioned in the patient report you will be able to provide other possible different diagnosis by understanding minute details of that report
Report:
{report_text}

Output format:
1. **Primary Diagnosis**: [Diagnosis]  
   Reasoning: [Detailed clinical rationale]  

2. **Secondary Possibility**: [Diagnosis]  
   Reasoning: [...]

3. **Less Likely but Consider**: [Diagnosis]  
   Reasoning: [...]

Also mention if any urgent/life-threatening conditions only if it is needed and you should must be ruled out and why.
""")

    try:
        prompt = detailed_prompt.format(report_text=st.session_state.full_text)

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.4,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        st.markdown("### 🔬 Diagnostic Reasoning")
        response = llm.invoke(prompt)
        st.markdown(response.content)

    except Exception as e:
        st.error(f"⚠️ Error during diagnosis reasoning: {e}")
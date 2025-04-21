from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

summarize_prompt = PromptTemplate.from_template("""
You are a professional medical assistant helping a healthcare provider. Summarize the key information from the patient report below.

Extract and clearly format:
- Patient Name
- Age / Gender
- Symptoms
- Diagnosis / Conditions
- Medications
- Dosage Instructions
- Precautions (Food, Sleep, Activity, Yoga)
- Follow-up Advice

Report:
{report_text}
""")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

chain: RunnableSequence = summarize_prompt | llm

def extract_important_details(text: str) -> str:
    try:
        response = chain.invoke({"report_text": text})
        return response.content
    except Exception as e:
        return f"⚠️ Error during summarization: {e}"
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise EnvironmentError("❌ GOOGLE_API_KEY not found in .env")

# Define the prompt template using LangChain
summarize_prompt = PromptTemplate.from_template("""
You are a medical assistant. Given the text extracted from a patient's health report, extract key structured information.

Extract the following:
- Patient Name (if available)
- Age / Gender (if mentioned)
- Symptoms
- Diagnosis / Conditions
- Medications
- Dosage Instructions
- Precautions (Food, Sleep, Physical Activity)
- Follow-up Advice

Text:
{report_text}

Format your output clearly in bullet points or sections.
""")

# Initialize Gemini LLM using LangChain wrapper
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",  # ✅ Picked from available list
    temperature=0.3,
    google_api_key=api_key
)

# Combine the prompt and LLM into a LangChain RunnableSequence
chain: RunnableSequence = summarize_prompt | llm

def extract_important_details(text: str) -> str:
    """
    Extract structured information from patient report text using Gemini + LangChain prompt template.
    """
    try:
        response = chain.invoke({"report_text": text})
        return response.content
    except Exception as e:
        return f"⚠️ Error during summarization: {e}"
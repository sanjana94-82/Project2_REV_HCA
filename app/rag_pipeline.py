# app/rag_pipeline.py
import sys
import os
from dotenv import load_dotenv

# Add the project root directory (one level up from /app)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
load_dotenv()

from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.embed_store import get_vectorstore

def get_rag_chain():
    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever()

    # Load Gemini API key from environment variable
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",  # ensure proper model naming
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Add chain_type to resolve input key error
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # Use "stuff" for full document context in response
        return_source_documents=False
    )
    return qa_chain

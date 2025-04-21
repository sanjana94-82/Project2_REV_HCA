import os
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from utils.embed_store import get_vectorstore

def get_rag_chain():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

    # Enhanced system prompt
    system_prompt = """
You are a highly intelligent, compassionate medical assistant helping healthcare providers.
You are given access to a patient's uploaded report, along with relevant context.
Even if the report is missing specific details, use your medical knowledge to give helpful, safe, and context-aware suggestions.

- Consider diagnosis, symptoms, and medications
- Include reasoning in your answers
- If something is not directly stated, infer safe general advice
- If unsure, give a balanced recommendation and suggest checking with the physician

Patient report content:
{context}

Provider question:
{question}
"""

    prompt = PromptTemplate(
        template=system_prompt,
        input_variables=["context", "question"]
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return rag_chain

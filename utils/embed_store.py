# utils/embed_store.py

import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
CHROMA_DIR = "data/chroma_db"

def store_embeddings(text, source_name="uploaded_report"):
    """Splits text into chunks, generates embeddings, and stores them in ChromaDB."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])

    # Tag source for future filtering or tracing
    for doc in docs:
        doc.metadata["source"] = source_name

    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    # db.persist()
    return f"Stored {len(docs)} chunks to ChromaDB."

def get_vectorstore(persist_directory="vectorstore"):
    # Create embedding model using Gemini
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # Load the Chroma vector store from the persistence directory
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    return vectordb

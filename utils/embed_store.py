import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
CHROMA_DIR = "data/chroma_db"

def store_embeddings(text, source_name="uploaded_report"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
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
    return f"Stored {len(docs)} chunks to ChromaDB."

def get_vectorstore(persist_directory=CHROMA_DIR):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

#embed_store.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

def store_embeddings(text, source_name="uploaded_report", persist_directory="data/chroma_db/default"):
    # Split the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    for doc in docs:
        doc.metadata["source"] = source_name

    # Set up embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Save to Chroma
    os.makedirs(persist_directory, exist_ok=True)
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    db.persist()
    return f"Stored {len(docs)} chunks to ChromaDB at {persist_directory}"

def get_vectorstore(persist_directory="data/chroma_db/default"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

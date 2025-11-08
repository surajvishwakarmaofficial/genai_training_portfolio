import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from src.utils.llm_helper import get_embeddings_model # Import from your utils

def process_pdfs(uploaded_files, embeddings_model, chroma_persist_dir="./chroma_db"):
    """Processes uploaded PDF files, splits them, and creates a Chroma vector store."""
    if not uploaded_files:
        st.warning("Please upload at least one PDF file")
        return None
    
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        documents.extend(docs)
        os.unlink(tmp_file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings_model,
        persist_directory=chroma_persist_dir
    )
    st.success(f"✅ Processed {len(uploaded_files)} PDF(s) with {len(splits)} chunks!")
    return vectorstore

def scrape_website(website_url, embeddings_model, chroma_persist_dir="./chroma_db"):
    """Scrapes a website, splits content, and creates a Chroma vector store."""
    if not website_url:
        st.warning("Please enter a valid URL")
        return None
    
    loader = WebBaseLoader(website_url)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings_model,
        persist_directory=chroma_persist_dir
    )
    st.success(f"✅ Scraped website with {len(splits)} chunks!")
    return vectorstore


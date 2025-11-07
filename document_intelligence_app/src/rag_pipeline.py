# src/rag_pipeline.py
import os
import time
import pinecone
import streamlit as st
from langchain_pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import (
    PINECONE_API_KEY, PINECONE_METRIC, PINECONE_CLOUD, PINECONE_REGION,
    PINECONE_DEFAULT_INDEX, CONTEXT_SUFFIX,
    TEXT_SPLITTER_CHUNK_SIZE, TEXT_SPLITTER_CHUNK_OVERLAP
)
from src.utils import sanitize_filename_for_pinecone

def setup_pinecone_index(index_name: str, dimension: int):
    """
    Initializes Pinecone client and ensures the specified index exists.
    If the index doesn't exist, it creates a new one.
    """
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    
    if index_name not in [index.name for index in pc.list_indexes()]:
        st.info(f"Creating new Pinecone index: '{index_name}' with dimension {dimension}...")
        pc.create_index(
            name=index_name, 
            dimension=dimension,
            metric=PINECONE_METRIC,
            spec=pinecone.ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status.ready:
            time.sleep(1)
        st.success(f"Index '{index_name}' created successfully!")
    else:
        st.info(f"Using existing Pinecone index: '{index_name}'")
        
    return pc.Index(index_name)

def process_and_store_pdf(uploaded_file, embeddings, expected_dimension: int):
    """
    Processes an uploaded PDF file: loads, splits, and stores chunks in Pinecone.
    """
    temp_file_path = f"./{uploaded_file.name}"
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CHUNK_SIZE, 
        chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    
    # Determine the index name. Try to use a sanitized filename, fallback to default.
    proposed_index_name = sanitize_filename_for_pinecone(uploaded_file.name)
    
    # Check if a custom index can be created or exists. Otherwise, fall back to default.
    try:
        pinecone_index = setup_pinecone_index(proposed_index_name, expected_dimension)
        index_to_use = proposed_index_name
    except Exception as e:
        st.warning(f"Could not create or use custom index '{proposed_index_name}': {e}. Falling back to default index '{PINECONE_DEFAULT_INDEX}'.")
        pinecone_index = setup_pinecone_index(PINECONE_DEFAULT_INDEX, expected_dimension)
        index_to_use = PINECONE_DEFAULT_INDEX

    st.info(f"Storing {len(texts)} document chunks in Pinecone index '{index_to_use}'...")
    vector_store = Pinecone.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=index_to_use
    )
    st.success(f"Successfully stored {len(texts)} chunks in Pinecone index '{index_to_use}'.")
    
    os.remove(temp_file_path)
    
    return len(texts), vector_store, index_to_use

def ask_document(question: str, vector_store, chat_model, user_prompt: str):
    """
    Retrieves relevant documents from the vector store and asks the chat model.
    """
    retriever = vector_store.as_retriever()
    retrieved_docs = retriever.invoke(question)
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    print(context)
    full_prompt_template = user_prompt + CONTEXT_SUFFIX
    final_system_prompt = full_prompt_template.format(context=context)
    
    messages = [
        SystemMessage(content=final_system_prompt),
        HumanMessage(content=question),
    ]
    
    return chat_model.invoke(messages).content

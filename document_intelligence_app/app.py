# app.py
import streamlit as st
import os
from langchain_core.messages import HumanMessage
from src.config import (
    DEFAULT_USER_PROMPT, HF_EMBEDDING_DIMENSION, AZURE_EMBEDDING_DIMENSION,
    GOOGLE_EMBEDDING_DIMENSION
)
from src.llm_providers import (
    initialize_huggingface_components, 
    initialize_azure_openai_components, 
    initialize_google_genai_components
)
from src.rag_pipeline import process_and_store_pdf, ask_document
from src.components import (
    load_css, display_main_header, display_chat_message, 
    display_sidebar_header, display_uploaded_file_card, 
    display_provider_card, display_metric_card, display_status_box
)

# Load custom CSS
load_css()

# Main App Layout
display_main_header()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_components" not in st.session_state:
    st.session_state.rag_components = {} # Stores vector_store, chat_model, index_name

if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = DEFAULT_USER_PROMPT

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

# Sidebar with improved layout
with st.sidebar:
    display_sidebar_header("⚙️ Configuration")
    
    # Mode selection
    mode = st.radio(
        "Select Mode",
        ["📄 Document Q&A (RAG)", "💬 Direct Chat"],
        help="Choose whether to chat with your documents or directly with the AI"
    )
    
    rag_on = (mode == "📄 Document Q&A (RAG)")
    
    if rag_on:
        display_sidebar_header("🔧 RAG Settings")
        
        # Provider selection with cards
        st.markdown("**Embedding Model Provider**")
        embedding_model_provider = st.selectbox(
            "Select Provider",
            ["Hugging Face", "Azure OpenAI", "Google Gemini"],
            label_visibility="collapsed"
        )
        display_provider_card(embedding_model_provider)
        
        # File upload with styled container
        st.markdown("**Document Upload**")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            label_visibility="collapsed",
            help="Upload a PDF document to create a knowledge base"
        )
        
        if uploaded_file:
            display_uploaded_file_card(uploaded_file.name, uploaded_file.size)
            
            # Process button - only show when a new file is uploaded or not yet processed
            if st.session_state.current_file_name != uploaded_file.name or not st.session_state.document_processed:
                if st.button("🚀 Process Document", use_container_width=True):
                    with st.spinner("🔍 Setting up document processing..."):
                        try:
                            embeddings = None
                            chat_model = None
                            dimension = 0

                            if embedding_model_provider == "Hugging Face":
                                embeddings, chat_model = initialize_huggingface_components()
                                dimension = HF_EMBEDDING_DIMENSION
                            elif embedding_model_provider == "Azure OpenAI":
                                embeddings, chat_model = initialize_azure_openai_components()
                                dimension = AZURE_EMBEDDING_DIMENSION
                            else:  # Google Gemini
                                embeddings, chat_model = initialize_google_genai_components()
                                dimension = GOOGLE_EMBEDDING_DIMENSION
                            
                            with st.spinner(f"📄 Processing '{uploaded_file.name}' and storing in Pinecone..."):
                                num_chunks, vector_store, index_name_used = process_and_store_pdf(uploaded_file, embeddings, dimension)
                            
                            # Store in session state
                            st.session_state.rag_components = {
                                "vector_store": vector_store,
                                "chat_model": chat_model,
                                "index_name": index_name_used,
                                "num_chunks": num_chunks # Add num_chunks for status display
                            }
                            st.session_state.document_processed = True
                            st.session_state.current_file_name = uploaded_file.name
                            
                            st.success("✅ Document Processing Complete!")
                            # Rerun to update sidebar status immediately after processing
                            st.rerun() 
                            
                        except Exception as e:
                            st.error(f"❌ An error occurred during processing: {e}")
                            st.session_state.document_processed = False # Reset on error
        
        # Assistant instructions with expander
        with st.expander("🧠 Assistant Instructions", expanded=False):
            st.session_state.user_prompt = st.text_area(
                "Customize how the assistant should respond:",
                value=st.session_state.user_prompt,
                height=150,
                label_visibility="collapsed"
            )
            
    else: # Direct Chat Mode
        display_sidebar_header("💬 Chat Settings")
        llm_provider = st.selectbox("Select LLM Provider", ["Hugging Face", "Azure OpenAI", "Google Gemini"])
        # No processing button for direct chat

# Main content area - Display chat messages with custom styling
for message in st.session_state.messages:
    display_chat_message(message["role"], message["content"])

# Status panel
st.sidebar.markdown("### 📊 Status Panel")
    
if rag_on:
    if uploaded_file is not None:
        if st.session_state.document_processed and st.session_state.current_file_name == uploaded_file.name:
            display_status_box("✅ Knowledge Base Connected", "success")
            display_metric_card("Document Name", st.session_state.current_file_name)
            display_metric_card("Index Name", st.session_state.rag_components.get("index_name", "N/A"))
            display_metric_card("Chunks Stored", str(st.session_state.rag_components.get("num_chunks", "N/A")))
            display_metric_card("Index Status", "🟢 Active")
        else:
            display_status_box("⏳ Processing Required", "warning")
            if uploaded_file and st.session_state.current_file_name != uploaded_file.name:
                 display_metric_card("Document Name", uploaded_file.name)
                 display_metric_card("Status", "New file, needs processing")
            elif uploaded_file:
                 display_metric_card("Document Name", uploaded_file.name)
                 display_metric_card("Status", "Upload detected, click Process Document")
    else:
        display_status_box("📝 Please upload a document", "warning")
else:
    display_status_box("💬 Direct Chat Mode", "info")
    display_metric_card("Provider", llm_provider)

# Chat input at bottom
st.markdown("---")
input_col, button_col = st.columns([4, 1])

with input_col:
    prompt = st.chat_input("💬 Ask your question here...")

with button_col:
    if st.session_state.messages:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            # Optionally clear RAG components if you want to start fresh with a new doc
            # st.session_state.rag_components = {}
            # st.session_state.document_processed = False
            # st.session_state.current_file_name = None
            st.rerun()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                answer = ""
                if rag_on:
                    if st.session_state.document_processed and st.session_state.rag_components:
                        vs = st.session_state.rag_components["vector_store"]
                        cm = st.session_state.rag_components["chat_model"]
                        answer = ask_document(prompt, vs, cm, st.session_state.user_prompt)
                    else:
                        answer = "📝 Please upload and process a document to begin the RAG chat. Click the 'Process Document' button after uploading your PDF."
                else: # Direct Chat Mode
                    chat_model = None
                    if llm_provider == "Hugging Face":
                        _, chat_model = initialize_huggingface_components()
                    elif llm_provider == "Azure OpenAI":
                        _, chat_model = initialize_azure_openai_components()
                    else:  # Google Gemini
                        _, chat_model = initialize_google_genai_components()
                    
                    if chat_model:
                        answer = chat_model.invoke([HumanMessage(content=prompt)]).content
                    else:
                        answer = "Error: Chat model could not be initialized."
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f" An error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
import os
import streamlit as st
from dotenv import load_dotenv

# Import components and utilities
from src.components.data_processor import process_pdfs, scrape_website
from src.components.chat_interface import display_chat_history
from src.utils.llm_helper import get_embeddings_model, get_llm_model, create_conversation_chain, get_custom_prompt_template

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Student Assistant Chatbot",
    page_icon="🎓",
    layout="wide"
)

# Environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")


# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation' not in st.session_state:
    st.session_state.conversation = None

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🎓 Student Assistant Chatbot</h1>", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.divider()
    
    st.header("📚 Data Sources")
    
    data_source = st.radio(
        "Choose data source:",
        ["Upload PDF", "Scrape Website"]
    )
    
    embeddings_model = get_embeddings_model(AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT)

    if data_source == "Upload PDF":
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files containing college information"
        )
        
        if st.button("Process PDFs", type="primary"):
            with st.spinner("Processing PDFs..."):
                try:
                    st.session_state.vectorstore = process_pdfs(uploaded_files, embeddings_model)
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")
    
    else:  # Scrape Website
        website_url = st.text_input(
            "College Website URL",
            placeholder="https://example.edu",
            help="Enter the URL of your college website"
        )
        
        if st.button("Scrape Website", type="primary"):
            with st.spinner("Scraping website..."):
                try:
                    st.session_state.vectorstore = scrape_website(website_url, embeddings_model)
                except Exception as e:
                    st.error(f"Error scraping website: {str(e)}")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.conversation = None
        st.rerun()

# Main chat interface
st.header("💬 Chat with Your Student Assistant")

# Display chat history using the component
chat_container = st.container()
with chat_container:
    display_chat_history(st.session_state.chat_history)

# Chat input
user_question = st.chat_input("Ask me anything about your college...")

if user_question:
    if st.session_state.vectorstore is None:
        st.error("⚠️ Please upload a PDF or scrape a website first")
    else:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        with st.spinner("Thinking..."):
            try:
                if st.session_state.conversation is None:
                    llm = get_llm_model(DEPLOYMENT_NAME, AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT)
                    prompt_template = get_custom_prompt_template()
                    
                    st.session_state.conversation = create_conversation_chain(
                        llm,
                        st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                        prompt_template
                    )
                
                response = st.session_state.conversation({
                    "question": user_question
                })
                
                assistant_response = response['answer']
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Instructions
with st.expander("📖 How to Use"):
    st.markdown("""
    ### Getting Started:
    
    1. **Set Up .env**: Create a `.env` file in the root directory and add your Azure OpenAI credentials.
    2. **Install Dependencies**: `pip install -r requirements.txt`
    3. **Choose Data Source**:
       - **Upload PDF**: Upload PDF files containing college information
       - **Scrape Website**: Enter your college website URL to scrape data
    
    4. **Process Data**: Click the process button to convert data into vector embeddings
    
    5. **Start Chatting**: Ask questions about your college in the chat box below
    
    ### Example Questions:
    - What courses are offered?
    - Tell me about admission requirements
    - What facilities are available?
    - What is the fee structure?
    - Information about scholarships
    
    ### Run the Application:
    ```bash
    streamlit run src/app.py
    ```
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with LangChain 🦜🔗 and ChromaDB 💾</p>
</div>
""", unsafe_allow_html=True)


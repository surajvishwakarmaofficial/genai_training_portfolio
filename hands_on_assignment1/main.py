
import os
import time
import tempfile
import logging
from typing import Tuple, List, Optional
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv
import pinecone
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Configuration Management
@dataclass
class AppConfig:
    """Application configuration settings."""
    PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")
    AZURE_OPENAI_KEY: str = os.environ.get("AZURE_OPENAI_KEY", "")
    AZURE_OPENAI_ENDPOINT: str = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    DEFAULT_INDEX_NAME: str = "pdf-chatbot-index"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 20
    EMBEDDING_DIMENSION: int = 1536
    SIMILARITY_SEARCH_K: int = 3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF processing and vector store operations."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._validate_environment()
        self.pc = pinecone.Pinecone(api_key=config.PINECONE_API_KEY)
        self.embeddings = self._initialize_embeddings()
    
    def _validate_environment(self) -> None:
        """Validate that all required environment variables are set."""
        if not self.config.PINECONE_API_KEY:
            raise EnvironmentError("PINECONE_API_KEY not set in environment or .env")
        if not self.config.AZURE_OPENAI_KEY:
            raise EnvironmentError("AZURE_OPENAI_KEY not set in environment or .env")
        if not self.config.AZURE_OPENAI_ENDPOINT:
            raise EnvironmentError("AZURE_OPENAI_ENDPOINT not set in environment or .env")
    
    def _initialize_embeddings(self) -> AzureOpenAIEmbeddings:
        """Initialize Azure OpenAI embeddings."""
        return AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small",
            model="text-embedding-3-small",
            openai_api_type="azure",
            openai_api_key=self.config.AZURE_OPENAI_KEY,
            azure_endpoint=self.config.AZURE_OPENAI_ENDPOINT,
            openai_api_version="2023-05-15",
            chunk_size=2048
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, int]:
        """Extract text from PDF file."""
        try:
            reader = PdfReader(pdf_path)
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages)
            
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF file.")
            
            return text, len(pages)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into manageable chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len
        )
        return text_splitter.split_text(text)
    
    def create_or_get_index(self, index_name: str) -> None:
        """Create Pinecone index if it doesn't exist."""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if index_name not in existing_indexes:
            logger.info(f"Creating index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=self.config.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            while not self.pc.describe_index(index_name).status.ready:
                time.sleep(1)
            logger.info(f"Index {index_name} created successfully")
        else:
            logger.info(f"Index {index_name} already exists")
    
    def process_uploaded_pdf(self, uploaded_file, index_name: str) -> PineconeVectorStore:
        """Process uploaded PDF and create vector store."""
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        try:
            # Extract and process text
            text, page_count = self.extract_text_from_pdf(pdf_path)
            logger.info(f"PDF processed successfully. Total pages: {page_count}")
            
            # Split text into chunks
            texts = self.split_text_into_chunks(text)
            logger.info(f"Created {len(texts)} text chunks")
            
            # Create or get index
            self.create_or_get_index(index_name)
            
            # Create documents and upload to Pinecone
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
            chunks = text_splitter.create_documents([text])
            
            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                index_name=index_name,
                embedding=self.embeddings,
                pinecone_api_key=self.config.PINECONE_API_KEY
            )
            
            logger.info("PDF processing completed successfully")
            return vectorstore
            
        finally:
            # Clean up temporary file
            os.unlink(pdf_path)

class ChatManager:
    """Manages chat functionality and session state."""
    
    def __init__(self):
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state."""
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = None
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'processed' not in st.session_state:
            st.session_state.processed = False
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None
    
    def add_message(self, role: str, content: str) -> None:
        """Add message to chat history."""
        st.session_state.messages.append({"role": role, "content": content})
    
    def reset_chat(self) -> None:
        """Reset chat history."""
        st.session_state.messages = []
    
    def reset_application(self) -> None:
        """Reset entire application state."""
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.session_state.processed = False
        st.session_state.uploaded_file = None

def similarity_search(vectorstore: PineconeVectorStore, question: str, k: int = 3) -> List[str]:
    """Perform similarity search and return results."""
    try:
        results = vectorstore.similarity_search(question, k=k)
        logger.info(f"Similarity search completed for query: {question}")
        return [doc.page_content for doc in results]
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        raise

def setup_ui(chat_manager: ChatManager, pdf_processor: PDFProcessor) -> None:
    """Setup the Streamlit user interface."""
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF Chatbot")
    st.markdown("Upload a PDF and ask questions about it!")
    
    _render_sidebar(chat_manager, pdf_processor)
    _render_main_content(chat_manager)

def _render_sidebar(chat_manager: ChatManager, pdf_processor: PDFProcessor) -> None:
    """Render the sidebar components."""
    with st.sidebar:
        st.header("Configuration")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload your PDF document"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            st.success(f"File uploaded: {uploaded_file.name}")
        
        index_name = st.text_input(
            "Pinecone Index Name",
            value="pdf-chatbot-index",
            help="Name of the Pinecone index"
        )
        
        # Process PDF button
        process_disabled = st.session_state.uploaded_file is None or st.session_state.processed
        if st.button("Process PDF", disabled=process_disabled, type="primary"):
            _handle_pdf_processing(pdf_processor, index_name)
        
        # Application controls
        if st.session_state.processed:
            st.success("PDF processed successfully!")
            
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset Chat"):
                chat_manager.reset_chat()
                st.rerun()
        
        with col2:
            if st.button("New PDF"):
                chat_manager.reset_application()
                st.rerun()

def _handle_pdf_processing(pdf_processor: PDFProcessor, index_name: str) -> None:
    """Handle PDF processing with error handling."""
    try:
        with st.spinner("Processing PDF..."):
            vectorstore = pdf_processor.process_uploaded_pdf(
                st.session_state.uploaded_file, 
                index_name
            )
            st.session_state.vectorstore = vectorstore
            st.session_state.processed = True
            st.rerun()
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        st.error(f"Error processing PDF: {str(e)}")

def _render_main_content(chat_manager: ChatManager) -> None:
    """Render the main chat content area."""
    if not st.session_state.processed:
        if st.session_state.uploaded_file is None:
            st.info("Please upload a PDF file using the sidebar.")
        else:
            st.info("Click 'Process PDF' in the sidebar to process your uploaded PDF.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the PDF..."):
        _handle_user_query(chat_manager, prompt)

def _handle_user_query(chat_manager: ChatManager, prompt: str) -> None:
    """Handle user query and generate response."""
    chat_manager.add_message("user", prompt)
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching for relevant information..."):
            try:
                results = similarity_search(
                    st.session_state.vectorstore, 
                    prompt, 
                    k=AppConfig.SIMILARITY_SEARCH_K
                )
                
                # Format and display results
                formatted_results = _format_search_results(results)
                st.markdown(formatted_results)
                
                chat_manager.add_message("assistant", formatted_results)
                
            except Exception as e:
                error_msg = f"Error searching for information: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                chat_manager.add_message("assistant", error_msg)

def _format_search_results(results: List[str]) -> str:
    """Format search results for display."""
    if not results:
        return "No relevant information found."
    
    formatted = "**Relevant Information:**\n\n"
    for i, result in enumerate(results, 1):
        formatted += f"{i}. {result}\n\n"
    
    return formatted

def main():
    """Main application entry point."""
    try:
        config = AppConfig()
        pdf_processor = PDFProcessor(config)
        chat_manager = ChatManager()
        
        setup_ui(chat_manager, pdf_processor)
        
    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        st.error(f"Application initialization failed: {e}")

if __name__ == "__main__":
    main()
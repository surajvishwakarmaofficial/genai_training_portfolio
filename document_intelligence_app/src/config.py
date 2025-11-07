# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Model Settings
HF_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
HF_LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_LLM_TEMPERATURE = 0.5
HF_LLM_MAX_NEW_TOKENS = 512

AZURE_EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
AZURE_EMBEDDING_MODEL = "text-embedding-3-small"
AZURE_EMBEDDING_CHUNK_SIZE = 2048
AZURE_CHAT_TEMPERATURE = 0.5
AZURE_CHAT_API_VERSION = '2023-03-15-preview' # Update if needed

GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
GOOGLE_CHAT_MODEL = "gemini-2.0-flash"
GOOGLE_CHAT_TEMPERATURE = 0.5

# Pinecone Settings
PINECONE_DEFAULT_INDEX = "pdf-data"
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Embedding Dimensions
HF_EMBEDDING_DIMENSION = 768
AZURE_EMBEDDING_DIMENSION = 1536
GOOGLE_EMBEDDING_DIMENSION = 768

# Document Processing Settings
TEXT_SPLITTER_CHUNK_SIZE = 1000
TEXT_SPLITTER_CHUNK_OVERLAP = 100

# Prompts
DEFAULT_USER_PROMPT = "You are a helpful assistant. Your task is to answer the user's question based only on the following context. If the answer is not in the context, say 'I do not have enough information to answer that question.'"
CONTEXT_SUFFIX = "\n\nContext:\n{context}"
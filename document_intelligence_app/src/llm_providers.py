# src/llm_providers.py
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from src.config import (
    HUGGINGFACE_API_KEY, HF_EMBEDDING_MODEL, HF_LLM_REPO_ID, HF_LLM_TEMPERATURE, HF_LLM_MAX_NEW_TOKENS,
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_EMBEDDING_DEPLOYMENT, AZURE_EMBEDDING_MODEL,
    AZURE_EMBEDDING_CHUNK_SIZE, DEPLOYMENT_NAME, AZURE_CHAT_TEMPERATURE, AZURE_CHAT_API_VERSION,
    GOOGLE_API_KEY, GOOGLE_EMBEDDING_MODEL, GOOGLE_CHAT_MODEL, GOOGLE_CHAT_TEMPERATURE
)

def initialize_huggingface_components():
    """Initializes Hugging Face embeddings and chat model."""
    embeddings = HuggingFaceEndpointEmbeddings(model=HF_EMBEDDING_MODEL, huggingfacehub_api_token=HUGGINGFACE_API_KEY)
    llm = HuggingFaceEndpoint(
        repo_id=HF_LLM_REPO_ID, task="text-generation", temperature=HF_LLM_TEMPERATURE,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY, max_new_tokens=HF_LLM_MAX_NEW_TOKENS
    )
    return embeddings, ChatHuggingFace(llm=llm)

def initialize_azure_openai_components():
    """Initializes Azure OpenAI embeddings and chat model."""
    embeddings = AzureOpenAIEmbeddings(
        deployment=AZURE_EMBEDDING_DEPLOYMENT,
        model=AZURE_EMBEDDING_MODEL,
        openai_api_type="azure",
        openai_api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version="2023-05-15", # This might need to be adjusted based on your Azure deployment
        chunk_size=AZURE_EMBEDDING_CHUNK_SIZE
    )
    chat_model = AzureChatOpenAI(
        azure_deployment=DEPLOYMENT_NAME, 
        temperature=AZURE_CHAT_TEMPERATURE, 
        api_key=AZURE_OPENAI_KEY, 
        api_version=AZURE_CHAT_API_VERSION, 
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    return embeddings, chat_model

def initialize_google_genai_components():
    """Initializes Google Generative AI embeddings and chat model."""
    embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
    chat_model = ChatGoogleGenerativeAI(model=GOOGLE_CHAT_MODEL, temperature=GOOGLE_CHAT_TEMPERATURE, google_api_key=GOOGLE_API_KEY)
    return embeddings, chat_model
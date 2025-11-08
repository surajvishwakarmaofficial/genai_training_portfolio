import os
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_classic import conversational_retrieval
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate


def get_embeddings_model(api_key, endpoint):
    """Initializes and returns the Azure OpenAI Embeddings model."""
    return AzureOpenAIEmbeddings(
        deployment="text-embedding-3-small",
        model="text-embedding-3-small",
        openai_api_type="azure",
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        openai_api_version="2023-05-15",
        chunk_size=2048
    )

def get_llm_model(deployment_name, api_key, endpoint):
    """Initializes and returns the Azure Chat OpenAI LLM."""
    return AzureChatOpenAI(
        azure_deployment=deployment_name,
        temperature=0.5,
        api_key=api_key,
        api_version='2023-03-15-preview',
        azure_endpoint=endpoint
    )

def create_conversation_chain(llm, vectorstore_retriever, prompt_template):
    """Creates and returns the conversational retrieval chain."""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    return conversational_retrieval.from_llm(
        llm=llm,
        retriever=vectorstore_retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])},
        return_source_documents=True
    )

def get_custom_prompt_template():
    """Returns the custom prompt template string."""
    return """You are a helpful student assistant. Your role is to help students by answering their questions about college information, courses, admissions, facilities, and any other academic-related queries.

Use the following context to answer the student's question. If you don't know the answer based on the context, say so politely and offer to help with related information.

Context: {context}

Student Question: {question}

Helpful Answer:"""



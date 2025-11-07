# src/components.py
import streamlit as st
import os

def load_css(file_path="ui/styles.css"):
    """Loads custom CSS from a file."""
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found at {file_path}. Using default Streamlit styles.")

def display_main_header():
    """Displays the main application header."""
    st.markdown('<h1 class="main-header">📚 Document Intelligence Assistant</h1>', unsafe_allow_html=True)

def display_chat_message(role: str, content: str):
    """Displays a chat message with custom styling."""
    if role == "user":
        st.markdown(f'<div class="chat-user"><strong>You:</strong> {content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-assistant"><strong>Assistant:</strong> {content}</div>', unsafe_allow_html=True)

def display_sidebar_header(title: str):
    """Displays a stylized sidebar header."""
    st.markdown(f'<div class="sidebar-header">{title}</div>', unsafe_allow_html=True)

def display_uploaded_file_card(file_name: str, file_size: int):
    """Displays a card for the uploaded file."""
    st.markdown(f"""
    <div class="uploaded-file">
        📄 <strong>{file_name}</strong><br>
        <small>Size: {file_size / 1024:.1f} KB</small>
    </div>
    """, unsafe_allow_html=True)

def display_provider_card(provider_name: str):
    """Displays a card for the selected LLM provider."""
    provider_icons = {
        "Hugging Face": "🤗",
        "Azure OpenAI": "🔷", 
        "Google Gemini": "🔮"
    }
    icon = provider_icons.get(provider_name, "❓")
    st.markdown(f"""
    <div class="provider-card">
        <strong>{icon} {provider_name}</strong>
    </div>
    """, unsafe_allow_html=True)

def display_metric_card(title: str, value: str):
    """Displays a generic metric card."""
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <strong>{title}</strong><br>
        {value}
    </div>
    """, unsafe_allow_html=True)

def display_status_box(message: str, box_type: str = "info"):
    """Displays a styled status box (info, success, warning)."""
    if box_type == "success":
        st.sidebar.markdown(f'<div class="success-box">{message}</div>', unsafe_allow_html=True)
    elif box_type == "warning":
        st.sidebar.markdown(f'<div class="warning-box">{message}</div>', unsafe_allow_html=True)
    else: # default to info
        st.sidebar.markdown(f'<div class="info-box">{message}</div>', unsafe_allow_html=True)
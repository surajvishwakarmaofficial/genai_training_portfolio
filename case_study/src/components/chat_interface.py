import streamlit as st

def display_chat_history(chat_history):
    """Displays the chat history in the Streamlit UI."""
    for message in chat_history:
        if message["role"] == "user":
            st.markdown(f"<div class='chat-message user-message'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message assistant-message'><strong>Assistant:</strong> {message['content']}</div>", unsafe_allow_html=True)

            
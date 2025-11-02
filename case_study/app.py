"""
Career Counseling AI Assistant
Uses Google Gemini AI to provide career guidance through conversational AI.
"""

import os
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


class CareerCounselorAI:
    """AI-powered career counseling assistant."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """Initialize the career counselor AI.
        
        Args:
            model_name: Name of the Gemini model to use
        """
        
        self.model_name = model_name
        self.api_key = self._get_api_key()
        self.chain = self._setup_chain()
    
    def _get_api_key(self) -> str:
        """Safely retrieve API key from environment variables."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "Enter your api Key":
            raise ValueError(
                "Google API key not found. "
                "Please set GOOGLE_API_KEY in your environment variables or .env file"
            )
        return api_key
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the chat prompt template for career counseling."""
        
        system_message = """You are an AI career counselor assistant that helps users with their career queries. 
Your responsibilities include:
- Asking clarifying questions to understand user career goals
- Recommending suitable roles and learning roadmaps  
- Providing personalized career guidance based on individual aspirations
- Tracking market trends and suggesting relevant opportunities

Start by understanding the user's background, interests, and goals through conversation."""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", "{question}")
        ])
    
    def _setup_chain(self) -> RunnableSequence:
        """Set up the LangChain processing pipeline."""
        
        prompt = self._create_prompt_template()
        model = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.7  # Balanced creativity and consistency
        )
        parser = StrOutputParser()
        
        return prompt | model | parser
    
    def get_guidance(self, question: str) -> str:
        """Get career guidance for a specific question.
        
        Args:
            question: The career-related question to ask
            
        Returns:
            AI-generated career guidance response
        """
        try:
            response = self.chain.invoke({"question": question})
            return response.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def start_conversation(self) -> None:
        """Start an interactive career counseling session."""
        print(" Career Counselor AI: Hello! I'm here to help with your career questions.")
        print("Tell me about your career aspirations or ask me anything!\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\n Career Counselor AI: Thank you for chatting! Wishing you success in your career journey! ")
                    break
                
                if not user_input:
                    print(" Career Counselor AI: Please tell me more about your career questions!")
                    continue
                
                print("\n Career Counselor AI: ", end="")
                response = self.get_guidance(user_input)
                print(response + "\n")
                
            except KeyboardInterrupt:
                print("\n\n Career Counselor AI: Session ended. Good luck with your career!")
                break
            except Exception as e:
                print(f"\n Error: {str(e)}")


def main():
    """Main function to demonstrate the career counselor AI."""
    try:
        counselor = CareerCounselorAI()
        
        # Example single question
        sample_question = "I'm a computer science graduate with 2 years of experience in web development. I'm interested in AI and machine learning. What career path would you recommend?"
        
        print(" Career Counselor AI Demonstration")
        print("=" * 50)
        
        response = counselor.get_guidance(sample_question)
        print(f"Question: {sample_question}")
        print(f"Response: {response}")
        print("=" * 50)
        
        # Uncomment the line below for interactive conversation
        # counselor.start_conversation()
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")


if __name__ == "__main__":
    main()
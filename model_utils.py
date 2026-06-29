import os
import groq
import streamlit as st # Import streamlit to access st.secrets
from typing import Dict, Any, List
import logging
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3

# Configuration from environment variables (using st.secrets for Streamlit Cloud compatibility)
# Fallback to os.getenv for local development if st.secrets is not available (e.g., not running via streamlit run)
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    LOGGING_LEVEL = st.secrets.get("LOGGING_LEVEL", "INFO").upper()
    DEFAULT_MODEL = st.secrets.get("DEFAULT_MODEL", "llama3-8b-8192")
    DEFAULT_TEMPERATURE = float(st.secrets.get("DEFAULT_TEMPERATURE", 0.7))
    DEFAULT_MAX_TOKENS = int(st.secrets.get("DEFAULT_MAX_TOKENS", 500))
except Exception: # st.secrets not available, likely local development
    from dotenv import load_dotenv
    load_dotenv() # Load .env for local development
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO").upper()
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3-8b-8192")
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.7))
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", 500))


# LLM Model Configuration
AVAILABLE_MODELS = {
    "llama-3.1-8b-instant": "LLama 3.1 8B - Fast model with 128K context window",
    "llama-3.3-70b-versatile": "LLama 3.3 70B - Versatile model with 128K context window",
    "meta-llama/llama-4-scout-17b-16e-instruct": "LLama 4 Scout - Latest Llama 4 with 128K context",
    "deepseek-r1-distill-llama-70b": "DeepSeek R1 Distill Llama 70B - Distilled model with 128K context"
}

# Configure logging
logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)

# Groq API Key check
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables or Streamlit secrets.")
    raise ValueError("GROQ_API_KEY is not set.")

# Initialize Groq client
try:
    groq_client = groq.Client(api_key=GROQ_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    raise

def get_available_models() -> Dict[str, str]:
    """Return the dictionary of available models."""
    return AVAILABLE_MODELS


def get_model_response(
    query: str,
    context: List[str],
    model_id: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> str:
    """
    Get a response from the Groq LLM model based on the query and context.
    
    Args:
        query: The user's question
        context: List of relevant document chunks
        model_id: The ID of the model to use
        temperature: Controls randomness (0-1)
        max_tokens: Maximum number of tokens in the response
        
    Returns:
        The model's response as a string
    """
    # Create system message
    system_message = """You are a friendly and helpful assistant 🤖 designed to answer questions about the documents you're given. Your goal is to provide a great user experience!

Here are your guidelines:
1.  **Be Friendly and Use Emojis:** Start with a warm greeting and use emojis to make your answers more engaging. For example, you could say, "Hello there! 👋 How can I help you with your documents today?"
2.  **Strictly Stick to the Documents:** Your answers MUST be based *only* on the information in the documents provided. Do not use any outside knowledge.
3.  **Handle Off-Topic Questions:** If the user asks a question that is not related to the content of the documents, you MUST NOT attempt to answer it. Instead, you should politely state that you can only answer questions about the provided documents. For example: "I can only answer questions about the documents you've uploaded. Is there anything in the documents I can help you with? 🤔"
4.  **Cite Your Sources:** When you answer, let the user know where you found the information. It helps build trust!
5.  **If You Don't Know, Say So:** If the answer isn't in the documents, it's perfectly fine to say, "I couldn't find an answer to that in the documents. 😔 Is there anything else I can help with?"
6.  **Keep it Conversational:** Use a friendly and natural tone. Think of it as a helpful chat with a friend.

Let's make this a great experience for the user! 🚀"""

    # Create user message with context
    user_message = f"""Information:
{str(context)}

Question: {query}"""

    try:
        logger.info(f"Querying model {model_id} with temperature {temperature} and max_tokens {max_tokens}.")
        # Call the Groq API
        response = groq_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        response_content = response.choices[0].message.content
        logger.info("Successfully received response from model.")
        return response_content
    
    except groq.APIError as e:
        logger.error(f"Groq API error: {e}")
        return f"Error from Groq API: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {str(e)}"

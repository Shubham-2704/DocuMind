import streamlit as st
import os
import uuid
import logging
import atexit
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import shutil
from document_processor import process_document, query_documents, delete_document
from model_utils import get_available_models, get_model_response

# Configuration from environment variables (using st.secrets for Streamlit Cloud compatibility)
# Fallback to os.getenv for local development if st.secrets is not available (e.g., not running via streamlit run)
try:
    LOGGING_LEVEL = st.secrets.get("LOGGING_LEVEL", "INFO").upper()
    DEFAULT_MODEL = st.secrets.get("DEFAULT_MODEL", "llama3-8b-8192")
    DEFAULT_TEMPERATURE = float(st.secrets.get("DEFAULT_TEMPERATURE", 0.7))
    DEFAULT_MAX_TOKENS = int(st.secrets.get("DEFAULT_MAX_TOKENS", 500))
    CHROMA_DB_PATH = st.secrets.get("CHROMA_DB_PATH", "chroma_db")
except Exception: # st.secrets not available or empty, likely local development
    from dotenv import load_dotenv
    load_dotenv() # Load .env for local development
    LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO").upper()
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3-8b-8192")
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.7))
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", 500))
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")

# Configure logging
logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="DocuMind: Your Intelligent Document Analyst",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = f"user_documents_{uuid.uuid4().hex}"
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL
    if "db_path" not in st.session_state:
        db_session_path = os.path.join(CHROMA_DB_PATH, f"session_{uuid.uuid4().hex}")
        os.makedirs(db_session_path, exist_ok=True)
        st.session_state.db_path = db_session_path

init_session_state()

# Register cleanup function to delete session-specific ChromaDB directory on exit
def cleanup_session_data():
    if "db_path" in st.session_state and os.path.exists(st.session_state.db_path):
        try:
            shutil.rmtree(st.session_state.db_path)
            logger.info(f"Cleaned up session data directory: {st.session_state.db_path}")
        except OSError as e:
            logger.error(f"Error removing session data directory {st.session_state.db_path}: {e}")

atexit.register(cleanup_session_data)

def delete_file(file_name):
    """Helper function to delete a file and its data."""
    with st.spinner(f"Deleting {file_name}..."):
        try:
            delete_document(file_name, st.session_state.collection_name, st.session_state.db_path)
            st.session_state.uploaded_files.remove(file_name)
            # Clear chat history if no files are left
            if not st.session_state.uploaded_files:
                st.session_state.chat_history = []
            st.success(f"Deleted {file_name} and its related data.")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting {file_name}: {e}")
            logger.error(f"Error deleting {file_name}: {e}")

# Title and description
st.title("🚀 DocuMind: Your Intelligent Document Analyst")
st.markdown(
    """
    Unlock the knowledge within your documents. Upload, ask, and get intelligent answers instantly.
    """
)

# Sidebar for settings and file uploads
with st.sidebar:
    st.header("Settings")
    
    # Model selection
    available_models = get_available_models()
    model_options = list(available_models.keys())
    
    selected_model_index = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
    selected_model = st.selectbox(
        "Select LLM Model",
        options=model_options,
        format_func=lambda x: f"{x} - {available_models[x].split(' - ')[0]}",
        index=selected_model_index
    )
    st.session_state.selected_model = selected_model
    
    st.markdown(f"**Model Description:** {available_models[selected_model]}")
    
    # Temperature slider
    temperature = st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=DEFAULT_TEMPERATURE, 
        step=0.1,
        help="Controls randomness: 0 = deterministic, 1 = creative"
    )
    
    # Max tokens slider
    max_tokens = st.slider(
        "Max Response Length", 
        min_value=100, 
        max_value=2000, 
        value=DEFAULT_MAX_TOKENS, 
        step=100,
        help="Maximum number of tokens in the response"
    )
    
    st.divider()
    
    # Document upload section
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    try:
                        chunks_count = process_document(file, st.session_state.collection_name, db_path=st.session_state.db_path)
                        if file.name not in st.session_state.uploaded_files:
                            st.session_state.uploaded_files.append(file.name)
                        status_text.text(f"✅ {file.name} processed: {chunks_count} chunks extracted")
                        logger.info(f"Successfully processed {file.name}")
                    except Exception as e:
                        status_text.text(f"❌ Error processing {file.name}: {str(e)}")
                        logger.error(f"Error processing {file.name}: {e}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ All documents processed!")
                st.success(f"Successfully processed {len(uploaded_files)} documents")

    st.divider()

    # Display processed files in the sidebar
    if st.session_state.uploaded_files:
        st.subheader("Processed Files")
        for file_name in st.session_state.uploaded_files:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.text(f"📄 {file_name}")
            with col2:
                if st.button("❌", key=f"delete_sidebar_{file_name}"):
                    delete_file(file_name)
        st.divider()

    st.caption("Note: Data is stored locally for this session.")

# Display uploaded files
if st.session_state.uploaded_files:
    st.subheader("Uploaded Documents")
    for file_name in st.session_state.uploaded_files:
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.text(f"📄 {file_name}")
        with col2:
            if st.button("❌", key=f"delete_main_{file_name}"):
                delete_file(file_name)
else:
    st.info("No documents uploaded yet. Please upload documents using the sidebar.")

# Chat interface
if st.session_state.uploaded_files:
    st.subheader("Ask Questions About Your Documents")

    # Input for user question
    user_question = st.text_input("Your question:")

    # Submit button
    if st.button("Submit Question") and user_question:
        with st.spinner("Searching documents and generating answer..."):
            context = None
            try:
                results = query_documents(user_question, st.session_state.collection_name, db_path=st.session_state.db_path)
                if results and results.get("documents"):
                    context = results["documents"][0]
                else:
                    st.warning("No relevant information found in the documents for your query.")
            except Exception as e:
                st.error(f"Error querying documents: {e}")
                logger.error(f"Error querying documents: {e}")

            try:
                answer = get_model_response(
                    query=user_question,
                    context=context,
                    model_id=st.session_state.selected_model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                st.session_state.chat_history.append({"question": user_question, "answer": answer})
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                logger.error(f"Error generating answer: {e}")
else:
    st.info("Please upload and process at least one document to start the chat.")

# Display chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for i, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"**Question {i+1}:** {chat['question']}")
        st.markdown(f"**Answer:** {chat['answer']}")
        st.divider()

# Footer
st.markdown("---")

import os
import io
from typing import List, Dict, Any
import docx2txt
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import logging
from dotenv import load_dotenv
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3

load_dotenv()

# Configuration from environment variables
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO").upper()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")

# Configure logging
logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file content using PyMuPDF for speed."""
    text = ""
    with fitz.open(stream=file_content, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text() or ""
    return text


def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file content."""
    return docx2txt.process(io.BytesIO(file_content))


def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from TXT file content."""
    return file_content.decode('utf-8')


def extract_text(file_content: bytes, file_name: str) -> str:
    """Extract text based on file extension."""
    file_extension = os.path.splitext(file_name)[1].lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_content)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_content)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_content)
    else:
        logger.warning(f"Unsupported file format: {file_extension}")
        raise ValueError(f"Unsupported file format: {file_extension}")


def get_chroma_client(db_path: str = CHROMA_DB_PATH):
    """
    Get a ChromaDB client.
    In a production environment, you would use HttpClient to connect to a remote ChromaDB server.
    For local development, we use PersistentClient.
    """
    # Example for production:
    # return chromadb.HttpClient(host=os.getenv("CHROMA_DB_HOST"), port=int(os.getenv("CHROMA_DB_PORT")))
    return chromadb.PersistentClient(path=db_path)


def process_document(uploaded_file, collection_name: str, db_path: str) -> int:
    """Process uploaded document and store in ChromaDB."""
    try:
        file_content = uploaded_file.getvalue()
        file_name = uploaded_file.name
        
        logger.info(f"Processing document: {file_name}")
        
        # Extract text from the document
        document_text = extract_text(file_content, file_name)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_text(document_text)
        logger.info(f"Split document into {len(chunks)} chunks.")
        
        # Connect to ChromaDB
        chroma_client = get_chroma_client(db_path)
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        # Add document chunks to ChromaDB
        documents = []
        ids = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"{file_name}_chunk_{i}")
            metadatas.append({"source": file_name})
        
        # Batch upsert for efficiency
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            collection.upsert(
                documents=documents[i:i+batch_size],
                ids=ids[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
        
        logger.info(f"Successfully processed and stored {len(chunks)} chunks from {file_name}.")
        return len(chunks)
    
    except Exception as e:
        logger.error(f"Error processing document {uploaded_file.name}: {e}")
        raise


def query_documents(query_text: str, collection_name: str, db_path: str, n_results: int = 4) -> Dict[str, Any]:
    """Query the document collection for relevant chunks."""
    try:
        logger.info(f"Querying collection '{collection_name}' with query: '{query_text}'")
        chroma_client = get_chroma_client(db_path)
        collection = chroma_client.get_collection(name=collection_name)
        
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        logger.info(f"Found {len(results.get('documents', []))} results.")
        return results
    
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        return {}


def delete_document(file_name: str, collection_name: str, db_path: str):
    """Delete a document and its chunks from the collection."""
    try:
        logger.info(f"Deleting document '{file_name}' from collection '{collection_name}'")
        chroma_client = get_chroma_client(db_path)
        collection = chroma_client.get_collection(name=collection_name)
        
        # Delete chunks associated with the file
        collection.delete(where={"source": file_name})
        
        logger.info(f"Successfully deleted document '{file_name}' from collection.")
    
    except Exception as e:
        logger.error(f"Error deleting document {file_name}: {e}")
        raise

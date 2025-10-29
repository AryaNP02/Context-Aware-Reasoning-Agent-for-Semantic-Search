"""
Configuration module for RAG Application
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""
    
    # API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # Document Configuration
    DOCS_DIRECTORY = os.getenv("DOCS_DIRECTORY", "./Docs/")
    
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
    
    # Application Configuration
    MAX_RESULTS = int(os.getenv("MAX_RESULTS_PER_QUERY", "4"))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # RAG Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    SEPARATOR = "\n"
    
    @staticmethod
    def validate():
        """Validate required configuration"""
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in .env file.")
        
        # Create docs directory if it doesn't exist
        os.makedirs(Config.DOCS_DIRECTORY, exist_ok=True)
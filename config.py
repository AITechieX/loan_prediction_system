"""
Configuration module for Loan Prediction System.
Handles environment variable loading and provides centralized configuration.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class to manage environment variables."""
    
    # Groq API Configuration
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL_MAIN = os.getenv('GROQ_MODEL_MAIN', 'llama-3.3-70b-versatile')
    GROQ_MODEL_TEST = os.getenv('GROQ_MODEL_TEST', 'llama-3.3-70b-versatile')
    GROQ_TEMPERATURE = float(os.getenv('GROQ_TEMPERATURE', '0.3'))
    GROQ_MAX_TOKENS = int(os.getenv('GROQ_MAX_TOKENS', '1000'))
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present."""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required")
        return True

# Validate configuration on import
Config.validate_config()

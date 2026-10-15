from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# Vector Store Configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

# Ensure the Chroma directory exists
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# Embedding model
Embedding_Model = "BAAI/bge-small-en"

# Weather API key
Weather_API_KEY = os.getenv("WeatherAPI_KEY")

MODEL_Config = {
    "classifier": {
        "provider": "deepseek",
        "model_id": "us.deepseek.r1-v1:0"
    },
    "researcher": {
        "provider": "mistral",
        "model_id": "mistral.mixtral-8x7b-instruct-v0:1"
    },
    "weather": {
        "provider": "meta",
        "model_id": "us.meta.llama3-1-8b-instruct-v1:0"
    },
    "writer": {
        "provider": "meta",
        "model_id": "us.meta.llama3-3-70b-instruct-v1:0"
    },
    "critic": {
        "provider": "meta",
        "model_id": "us.meta.llama3-3-70b-instruct-v1:0"
    }
}

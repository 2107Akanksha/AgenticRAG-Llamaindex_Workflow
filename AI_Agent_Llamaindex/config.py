import os
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings
import time
from datetime import datetime, timedelta
# Load environment variables
load_dotenv()

GEMINI_API_KEY=os.getenv('GOOGLE_API_KEY')
COHERE_API_KEY=os.getenv('COHERE_API_KEY')

class RateLimiter:
    """Rate limiter for Gemini API calls."""
    def __init__(self):
        self.requests = []
        self.max_requests_per_minute = 60  # Free tier limit
        self.max_requests_per_day = 60 * 24  # Free tier daily limit
        self.window_size = 60  # 60 seconds

    def can_make_request(self) -> bool:
        now = datetime.now()
        # Remove old requests
        self.requests = [t for t in self.requests if now - t < timedelta(seconds=self.window_size)]
        
        # Check minute limit
        if len(self.requests) >= self.max_requests_per_minute:
            return False
            
        # Check daily limit
        daily_requests = len([t for t in self.requests if now - t < timedelta(days=1)])
        if daily_requests >= self.max_requests_per_day:
            return False
            
        return True

    def add_request(self):
        self.requests.append(datetime.now())

    def wait_if_needed(self):
        while not self.can_make_request():
            time.sleep(1)
        self.add_request()




# Memory Configuration
MEMORY_CONFIG = {
    'token_limit': int(os.getenv('MEMORY_TOKEN_LIMIT', '40000')),
    'chat_history_token_ratio': float(os.getenv('CHAT_HISTORY_TOKEN_RATIO', '0.7')),
    'token_flush_size': int(os.getenv('TOKEN_FLUSH_SIZE', '3000')),
}

# Global rate limiter instance
rate_limiter = RateLimiter()

# Initialize components
def init_components():
    """Initialize LLM, embedding model, and vector store."""
    # Initialize LLM with rate limiting
    class RateLimitedGemini(Gemini):
        def complete(self, *args, **kwargs):
            rate_limiter.wait_if_needed()
            return super().complete(*args, **kwargs)

    llm = RateLimitedGemini(
        model="models/gemini-1.5-flash",
        temperature=0.7,
    )

    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Initialize Chroma client and vector store
    db_path = os.getenv('VECTOR_STORE_DIR', './data/vector_store')
    os.makedirs(db_path, exist_ok=True)  # Ensure the directory exists
    
    # Create Chroma client with explicit settings
    chroma_client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(
            anonymized_telemetry=False,  # Disable telemetry
            is_persistent=True
        )
    )
    
    # Create or get collection
    collection = chroma_client.get_or_create_collection(name="memory_store")
    
    # Initialize vector store with the collection
    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    return llm, embed_model, vector_store, rate_limiter

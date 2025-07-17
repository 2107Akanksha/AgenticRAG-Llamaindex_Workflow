from typing import Optional, List, Any
from llama_index.core.memory import Memory
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding

def create_memory_config(
    session_id: str,
    llm: LLM,
    vector_store: Any,
    embed_model: BaseEmbedding,
    token_limit: int = 40000,
    chat_history_token_ratio: float = 0.7,
    token_flush_size: int = 3000,
) -> Memory:
    """
    Create a memory configuration with default settings.
    """
    return Memory.from_defaults(
        session_id=session_id,
        token_limit=token_limit,
        chat_history_token_ratio=chat_history_token_ratio,
        token_flush_size=token_flush_size,
    ) 
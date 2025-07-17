from typing import Dict, List, Optional, Any
import logging
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.memory import Memory
from memory.memory_config import create_memory_config

class MemoryAgent:
    """
    Handles memory management using LlamaIndex's Memory class for both short-term and long-term storage.
    """
    def __init__(
        self,
        llm: LLM,
        vector_store: Any,
        embed_model: BaseEmbedding,
        token_limit: int = 40000,
        chat_history_token_ratio: float = 0.7,
        token_flush_size: int = 3000,
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.token_limit = token_limit
        self.chat_history_token_ratio = chat_history_token_ratio
        self.token_flush_size = token_flush_size
        self.sessions: Dict[str, Memory] = {}
        self.logger = logging.getLogger(__name__)

    def get_or_create_memory(self, session_id: str) -> Memory:
        """Get existing memory or create new one for session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = create_memory_config(
                session_id=session_id,
                llm=self.llm,
                vector_store=self.vector_store,
                embed_model=self.embed_model,
                token_limit=self.token_limit,
                chat_history_token_ratio=self.chat_history_token_ratio,
                token_flush_size=self.token_flush_size,
            )
        return self.sessions[session_id]

    async def add_messages(self, session_id: str, messages: List[ChatMessage]) -> None:
        """Add messages to memory."""
        memory = self.get_or_create_memory(session_id)
        await memory.aput_messages(messages)
        self.logger.debug(f"Added messages to session {session_id}")

    async def get_chat_history(
        self, 
        session_id: str, 
        messages: Optional[List[ChatMessage]] = None
    ) -> List[ChatMessage]:
        """Get chat history including both short-term and long-term memory."""
        memory = self.get_or_create_memory(session_id)
        return await memory.aget(messages=messages)

    def clear_session(self, session_id: str) -> None:
        """Clear a session's memory."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.debug(f"Cleared session {session_id}") 
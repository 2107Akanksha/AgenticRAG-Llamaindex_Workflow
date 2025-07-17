from typing import List, Dict, Optional, Union
from pathlib import Path
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.core.base.llms.types import TextBlock
import chromadb
import logging
import os
from llama_index.core.memory import Memory, StaticMemoryBlock, FactExtractionMemoryBlock, VectorMemoryBlock, InsertMethod
from llama_index.core.llms import ChatMessage

class VectorMemoryStore:
    """Vector store for persistent memory storage using LlamaIndex."""
    
    def __init__(
        self,
        persist_dir: str = "memory_store",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "memory_store"
    ):
        # Create persist directory if it doesn't exist
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client with persistence
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        
        # Initialize vector store with persistence
        self.vector_store = ChromaVectorStore(
            chroma_client=self.chroma_client,
            collection_name=collection_name
        )
        
        # Setup storage context and index
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = None
        
        # Initialize node parser for text splitting
        self.node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        
        self.logger = logging.getLogger(__name__)
        
    def store_fact(self, fact: str, metadata: Optional[Dict] = None) -> Optional[str]:
        """Store a fact in the vector store with automatic text splitting."""
        try:
            # Parse text into nodes
            nodes = self.node_parser.get_nodes_from_documents(
                [Document(text=fact, metadata=metadata or {})]
            )
            
            if not nodes:
                return None
                
            # Get node ID before inserting
            node_id = str(nodes[0].node_id)
            
            # Initialize index if needed
            if not self.index:
                self.index = VectorStoreIndex(
                    nodes,
                    storage_context=self.storage_context,
                    embed_model=self.embed_model
                )
            else:
                # Insert nodes into existing index
                self.index.insert_nodes(nodes)
                
            return node_id
            
        except Exception as e:
            self.logger.error(f"Error storing fact: {str(e)}")
            raise
        
    def query_facts(
        self,
        query: str,
        top_k: int = 3,
        similarity_cutoff: float = 0.7
    ) -> List[Dict]:
        """
        Query relevant facts from the vector store with similarity filtering.
        
        Args:
            query: The query string
            top_k: Maximum number of results to return
            similarity_cutoff: Minimum similarity score (0-1) for results
        """
        if not self.index:
            return []
            
        try:
            # Create query engine with similarity cutoff
            query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                node_postprocessors=[
                    lambda x: [n for n in x if n.score >= similarity_cutoff]
                ]
            )
            
            # Execute query
            response = query_engine.query(query)
            
            # Process and return results
            results = []
            for node in response.source_nodes:
                results.append({
                    'text': node.text,
                    'metadata': node.metadata,
                    'score': node.score if hasattr(node, 'score') else None
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error querying facts: {str(e)}")
            return []
            
    def delete_fact(self, doc_id: str) -> bool:
        """Delete a fact from the vector store."""
        try:
            if self.index:
                self.index.delete_ref_doc(doc_id)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting fact: {str(e)}")
            return False
            
    def clear_store(self) -> bool:
        """Clear all facts from the vector store."""
        try:
            if self.chroma_client:
                self.chroma_client.reset()
                self.index = None
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error clearing store: {str(e)}")
            return False 

class MemoryAgent:
    def __init__(self, session_id, llm, vector_store, embed_model, token_limit=40000):
        # Set up memory blocks
        blocks = [
            StaticMemoryBlock(
                name="core_info",
                static_content=[TextBlock(text="Core information about the system and its capabilities")],
                priority=0,
            ),
            FactExtractionMemoryBlock(
                name="extracted_info",
                llm=llm,
                max_facts=50,
                priority=1,
            ),
            VectorMemoryBlock(
                name="vector_memory",
                vector_store=vector_store,
                priority=2,
                embed_model=embed_model,
            ),
        ]
        self.memory = Memory.from_defaults(
            session_id=session_id,
            token_limit=token_limit,
            memory_blocks=blocks,
            insert_method=InsertMethod.SYSTEM,
        )

    def add_messages(self, messages):
        self.memory.put_messages(messages)

    def get_context(self, messages=None):
        return self.memory.get(messages=messages) 
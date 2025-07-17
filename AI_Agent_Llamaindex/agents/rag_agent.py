from typing import List, Dict, Optional
import logging
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import Settings
from config import GEMINI_API_KEY, COHERE_API_KEY, rate_limiter
from .query_planner import QueryPlanner
import cohere
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from typing import List
from llama_index.core.schema import NodeWithScore

class CohereReranker(BaseNodePostprocessor):
    """Reranks search results using Cohere's reranking model."""
    
    def __init__(self, api_key: str, model: str = "rerank-english-v2.0", top_n: int = 3):
        """Initialize Cohere reranker.
        
        Args:
            api_key: Cohere API key
            model: Reranking model to use
            top_n: Number of top results to keep
        """
        self.co = cohere.Client(api_key)
        self.model = model
        self.top_n = top_n
        
    def postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Rerank nodes using Cohere's reranking model."""
        if not nodes:
            return nodes
            
        # Extract texts from nodes
        texts = [node.node.get_content() for node in nodes]
        
        # Get reranking scores from Cohere
        results = self.co.rerank(
            query=query_bundle.query_str,
            documents=texts,
            model=self.model,
            top_n=self.top_n
        )
        
        # Create mapping of text to original node
        text_to_node = {node.node.get_content(): node for node in nodes}
        
        # Reorder nodes based on Cohere scores
        reranked_nodes = []
        for result in results:
            node = text_to_node[result.document['text']]
            node.score = result.relevance_score
            reranked_nodes.append(node)
            
        return reranked_nodes

    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self.postprocess_nodes(nodes, query_bundle)

class RAGAgent:
    """
    Handles retrieval-augmented generation using vector DB and LLM.
    Includes query planning, decomposition, and hybrid search capabilities.
    """
    def __init__(self):
        class RateLimitedGemini(Gemini):
            def complete(self, *args, **kwargs):
                rate_limiter.wait_if_needed()
                return super().complete(*args, **kwargs)

        self.llm = RateLimitedGemini(
            model="models/gemini-1.5-flash",
            temperature=0.7,
        )
        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        self.logger = logging.getLogger(__name__)
        self.vector_index = None
        self.query_planner = QueryPlanner()

        # Fallback responses for common queries when API is unavailable
        self.fallback_responses = {
            "hello": "Hello! I'm an AI assistant. While I'm currently experiencing some technical limitations with my advanced capabilities, I'm still here to help as best I can.",}

    def initialize_retrievers(self, documents):
        """Initialize vector index retriever."""
        self.vector_index = VectorStoreIndex.from_documents(documents)
        
    def get_retriever(self):
        """Create a retriever with optimized parameters."""
        if not self.vector_index:
            raise ValueError("Vector index not initialized. Call initialize_retrievers first.")
            
        return VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=5
        )

    def transform_query(self, query: str) -> QueryBundle:
        """Transform the query to improve retrieval effectiveness."""
        transform_prompt = f"""Transform the following query to be more effective for retrieval.
        Add relevant synonyms and related terms while maintaining the original intent.
        Query: {query}
        Return the enhanced query."""
        
        response = self.llm.complete(transform_prompt)
        enhanced_query = response.text.strip()
        
        return QueryBundle(
            query_str=query,
            custom_embedding_strs=[enhanced_query]
        )

    def answer_sub_question(self, question: str, context: Optional[Dict] = None) -> str:
        """Answer a single sub-question using RAG."""
        if not self.vector_index:
            return "Error: Vector index not initialized"
            
        # Transform query for better retrieval
        query_bundle = self.transform_query(question)
        
        # Setup retriever and post-processing
        retriever = self.get_retriever()
        
        # Use both similarity cutoff and Cohere reranking
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY environment variable is not set")
        postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=0.7),
            CohereReranker(api_key=str(COHERE_API_KEY))
        ]
        
        # Create query engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=postprocessors
        )
        
        # Include context in the prompt if available
        context_str = ""
        if context:
            session_context = "\n".join([str(c) for c in context.get('session_context', [])])
            facts = "\n".join([f['text'] for f in context.get('long_term_facts', [])])
            context_str = f"\nSession Context:\n{session_context}\nRelevant Facts:\n{facts}"
            
        # Get response
        response = query_engine.query(
            f"""Answer the following question using the provided context.
            If the context doesn't contain relevant information, say so.
            
            Question: {question}{context_str}"""
        )
        
        return str(response).strip()

    def synthesize_answers(self, query: str, sub_answers: List[Dict[str, str]]) -> str:
        """Synthesize sub-answers into a coherent response."""
        answers_str = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" 
                               for qa in sub_answers])
        
        synthesis_prompt = f"""Synthesize the following question-answer pairs into a coherent response:
        Original Query: {query}
        
        {answers_str}
        
        Provide a comprehensive answer that addresses the original query."""
        
        response = self.llm.complete(synthesis_prompt)
        return response.text.strip()

    def get_fallback_response(self, query: str) -> str:
        """Provide a fallback response when API is unavailable."""
        # Convert query to lowercase for matching
        query_lower = query.lower()
        
        # Check for keyword matches in fallback responses
        for key, response in self.fallback_responses.items():
            if key in query_lower:
                return response
                
        return "I apologize, but I'm currently experiencing some technical limitations with my advanced capabilities. Please try again later or rephrase your question."

    def answer(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Main entry point for answering queries.
        Uses advanced query planning, caching, and multi-hop reasoning.
        """
        try:
            # Check cache first
            cached_response = self.query_planner.get_cached_response(query)
            if cached_response and len(cached_response) > 0:
                self.logger.info("Using cached response")
                return cached_response[0].get('answer', '')

            # Get execution plan
            execution_plan = self.query_planner.get_execution_plan(query, context)
            
            if execution_plan['type'] == 'cached':
                return execution_plan['plan']['answer']
                
            # Execute multi-hop reasoning steps
            sub_answers = []
            for step in execution_plan['plan']['steps']:
                # Get answer for this step
                answer = self.answer_sub_question(
                    step['optimized_query'],
                    {
                        **context,
                        'previous_steps': sub_answers
                    } if context else {'previous_steps': sub_answers}
                )
                
                sub_answers.append({
                    'question': step['question'],
                    'answer': answer,
                    'provides': step['provides']
                })
                
            # Synthesize final answer
            final_answer = self.synthesize_answers(query, sub_answers)
            
            # Update query history and cache
            self.query_planner.update_query_history(query, {
                'steps': sub_answers,
                'answer': final_answer
            })
            
            return final_answer
            
        except Exception as e:
            self.logger.error(f"Error in RAG pipeline: {str(e)}")
            if "429" in str(e) or "quota" in str(e).lower():
                return self.get_fallback_response(query)
            return f"I encountered an error processing your query. Please try again later." 
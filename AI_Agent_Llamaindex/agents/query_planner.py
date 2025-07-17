from typing import List, Dict, Optional, Set
import json
from datetime import datetime
import logging
from pathlib import Path
import hashlib
from functools import lru_cache

class QueryPlanner:
    """
    Advanced query planner with multi-hop reasoning, optimization, and caching capabilities.
    """
    def __init__(self, cache_dir: str = "data/query_cache"):
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.query_history: List[Dict] = []
        
    def _hash_query(self, query: str) -> str:
        """Generate a unique hash for the query."""
        return hashlib.md5(query.encode()).hexdigest()
        
    def _load_cache(self, query_hash: str) -> Optional[List[Dict]]:
        """Load cached response for a query."""
        cache_file = self.cache_dir / f"{query_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                # Check if cache is still valid (24 hours)
                if (datetime.now() - datetime.fromisoformat(cached['timestamp'])).days < 1:
                    return cached['response']
            except Exception as e:
                self.logger.error(f"Error loading cache: {str(e)}")
        return None
        
    def _save_cache(self, query_hash: str, response: List[Dict]):
        """Save response to cache."""
        cache_file = self.cache_dir / f"{query_hash}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'response': response
                }, f)
        except Exception as e:
            self.logger.error(f"Error saving cache: {str(e)}")
            
    def plan_multi_hop_query(self, query: str) -> List[Dict]:
        """
        Break down complex queries into a series of interconnected reasoning steps.
        Returns a list of query steps with dependencies.
        """
        # First check cache
        query_hash = self._hash_query(query)
        cached = self._load_cache(query_hash)
        if cached:
            return cached
            
        # Define reasoning steps prompt
        reasoning_prompt = f"""Break down the following query into a series of interconnected reasoning steps.
        For each step, identify:
        1. The specific question to answer
        2. What information it depends on from previous steps
        3. What new information it provides
        
        Query: {query}
        
        Return the steps in a structured format:
        Step 1: [specific question]
        Dependencies: none
        Provides: [information this step provides]
        
        Step 2: [specific question]
        Dependencies: [information needed from step 1]
        Provides: [information this step provides]
        ...
        """
        
        # TODO: Use LLM to generate reasoning steps
        # For now, using a simple decomposition
        steps = [
            {
                'step_number': 1,
                'question': query,
                'dependencies': [],
                'provides': 'direct answer'
            }
        ]
        
        # Save to cache
        self._save_cache(query_hash, steps)
        return steps
        
    def optimize_query(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Optimize query based on past interactions and available context.
        """
        if not context:
            context = {}
            
        # Analyze query history for similar queries
        similar_queries = self._find_similar_queries(query)
        
        # Build optimization context
        optimization_context = {
            'similar_queries': similar_queries,
            'user_context': context.get('user_context', {}),
            'session_context': context.get('session_context', [])
        }
        
        # TODO: Use LLM to optimize query based on context
        # For now, return original query
        return query
        
    def _find_similar_queries(self, query: str, threshold: float = 0.7) -> List[Dict]:
        """Find similar queries from history."""
        # TODO: Implement semantic similarity search
        return []
        
    @lru_cache(maxsize=1000)
    def get_cached_response(self, query: str) -> Optional[List[Dict]]:
        """Get cached response for exact query match."""
        query_hash = self._hash_query(query)
        return self._load_cache(query_hash)
        
    def update_query_history(self, query: str, response: Dict):
        """Update query history with new interaction."""
        self.query_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response
        })
        
        # Limit history size
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]
            
    def get_execution_plan(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Generate complete execution plan for a query.
        """
        # Check cache first
        cached = self.get_cached_response(query)
        if cached:
            return {
                'type': 'cached',
                'plan': {'steps': cached}
            }
            
        # Get multi-hop reasoning steps
        steps = self.plan_multi_hop_query(query)
        
        # Optimize each step
        optimized_steps = []
        for step in steps:
            optimized_query = self.optimize_query(step['question'], context)
            optimized_steps.append({
                **step,
                'optimized_query': optimized_query
            })
            
        return {
            'type': 'new',
            'plan': {
                'steps': optimized_steps,
                'context': context
            }
        } 
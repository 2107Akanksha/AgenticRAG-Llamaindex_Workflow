from typing import List, Dict, cast
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import numpy as np
import logging

class KeywordExtractor:
    """Extracts key terms and concepts from text using TF-IDF."""
    
    def __init__(self, max_features: int = 10):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.logger = logging.getLogger(__name__)
        
    def extract_keywords(self, text: str, top_k: int = 5) -> List[Dict[str, float]]:
        """Extract keywords from text using TF-IDF scores."""
        # Handle empty or invalid input
        if not text or not text.strip():
            return []
            
        # Fit and transform the text
        try:
            tfidf_matrix = cast(csr_matrix, self.vectorizer.fit_transform([text]))
            feature_names = self.vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]  
            
            # Get top-k keywords by score
            top_indices = np.argsort(scores)[-top_k:][::-1]
            keywords = [
                {
                    'keyword': feature_names[i],
                    'score': float(scores[i])
                }
                for i in top_indices if scores[i] > 0
            ]
            
            self.logger.debug(f"Extracted keywords from text: {keywords}")
            return keywords
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {str(e)}")
            return [] 
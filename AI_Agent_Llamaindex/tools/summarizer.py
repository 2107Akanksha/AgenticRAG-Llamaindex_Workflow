from typing import Optional
import logging
from llama_index.llms.gemini import Gemini
from config import GEMINI_API_KEY, rate_limiter

class Summarizer:
    """Generates concise summaries using Gemini."""
    
    def __init__(self):
        class RateLimitedGemini(Gemini):
            def complete(self, *args, **kwargs):
                rate_limiter.wait_if_needed()
                return super().complete(*args, **kwargs)

        self.llm = RateLimitedGemini(
            model="models/gemini-1.5-flash",
            temperature=0.7,
        )
        self.logger = logging.getLogger(__name__)
        
    def summarize(self, text: str, max_length: Optional[int] = None) -> str:
        """Generate a concise summary of the input text."""
        if not text or not text.strip():
            return ""
            
        length_constraint = f" in approximately {max_length} words" if max_length else ""
        
        prompt = f"""Generate a concise summary of the following text{length_constraint}.
        Focus on key points and maintain factual accuracy.
        
        Text to summarize:
        {text}
        
        Summary:"""
        
        try:
            response = self.llm.complete(prompt)
            summary = response.text.strip()
            
            self.logger.debug(f"Generated summary of length {len(summary.split())}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return "" 
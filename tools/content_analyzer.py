from typing import Dict, List
import logging
from llama_index.llms.gemini import Gemini
from config import GEMINI_API_KEY, rate_limiter

class ContentAnalyzer:
    """Analyzes document structure, themes, and sentiment."""
    
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
        
    def analyze_structure(self, text: str) -> Dict:
        """Analyze document structure (sections, paragraphs, etc.)."""
        if not text or not text.strip():
            return {'sections': [], 'structure_type': 'unknown'}
            
        prompt = """Analyze the structure of the following text.
        Identify main sections, their hierarchy, and document organization.
        Return a brief analysis focusing on document structure.
        
        Text to analyze:
        {text}
        
        Analysis:"""
        
        try:
            response = self.llm.complete(prompt)
            return {
                'structure_analysis': response.text.strip(),
                'sections': self._extract_sections(text)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing structure: {str(e)}")
            return {'sections': [], 'structure_type': 'error'}
            
    def analyze_themes(self, text: str) -> List[Dict]:
        """Identify main themes and topics in the text."""
        if not text or not text.strip():
            return []
            
        prompt = """Identify the main themes and topics in the following text.
        For each theme, provide a brief explanation of its significance.
        
        Text to analyze:
        {text}
        
        Themes:"""
        
        try:
            response = self.llm.complete(prompt)
            themes = [
                {'theme': line.strip(), 'confidence': 0.8}  # Simplified confidence
                for line in response.text.split('\n')
                if line.strip()
            ]
            return themes
        except Exception as e:
            self.logger.error(f"Error analyzing themes: {str(e)}")
            return []
            
    def _extract_sections(self, text: str) -> List[Dict]:
        """Helper method to extract document sections."""
        sections = []
        current_section = ""
        
        for line in text.split('\n'):
            if line.strip().isupper() or line.strip().endswith(':'):
                if current_section:
                    sections.append({
                        'title': current_section,
                        'content': line.strip()
                    })
                current_section = line.strip()
                
        return sections 
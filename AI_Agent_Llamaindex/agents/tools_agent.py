from typing import Dict, Any, Optional
import logging
from tools.keyword_extractor import KeywordExtractor
from tools.summarizer import Summarizer
from tools.content_analyzer import ContentAnalyzer

class ToolsAgent:
    """
    Hosts additional tools: keyword extraction, summarization, content analysis.
    """
    def __init__(self):
        self.keyword_extractor = KeywordExtractor()
        self.summarizer = Summarizer()
        self.content_analyzer = ContentAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        self.tools = {
            'extract_keywords': self.keyword_extractor.extract_keywords,
            'summarize': self.summarizer.summarize,
            'analyze_structure': self.content_analyzer.analyze_structure,
            'analyze_themes': self.content_analyzer.analyze_themes
        }

    def use_tool(self, tool_name: str, *args: Any, **kwargs: Any) -> Optional[Dict]:
        """Use a specific tool with given arguments."""
        if tool_name not in self.tools:
            self.logger.error(f"Tool {tool_name} not found")
            return None
            
        try:
            result = self.tools[tool_name](*args, **kwargs)
            self.logger.debug(f"Tool {tool_name} executed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error using tool {tool_name}: {str(e)}")
            return None 
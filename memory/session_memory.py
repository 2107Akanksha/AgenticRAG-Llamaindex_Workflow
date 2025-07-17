from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

class SessionMemory:
    """Manages short-term memory for conversation sessions."""
    
    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        
    def add_to_session(self, session_id: str, context: Dict) -> None:
        """Add context to a session's memory."""
        now = datetime.now()
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'contexts': [],
                'last_updated': now
            }
        
        self.sessions[session_id]['contexts'].append({
            'timestamp': now,
            **context
        })
        self.sessions[session_id]['last_updated'] = now
        
    def get_session_context(self, session_id: str, limit: int = 5) -> List[Dict]:
        """Get recent context from a session."""
        if session_id not in self.sessions:
            return []
            
        session = self.sessions[session_id]
        if datetime.now() - session['last_updated'] > self.session_timeout:
            del self.sessions[session_id]
            return []
            
        return session['contexts'][-limit:]
        
    def clear_session(self, session_id: str) -> None:
        """Clear a session's memory."""
        if session_id in self.sessions:
            del self.sessions[session_id] 
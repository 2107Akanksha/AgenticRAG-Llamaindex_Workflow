from typing import Dict, Optional, List, Any
import logging
import uuid
from agents.memory_agent import MemoryAgent
from agents.rag_agent import RAGAgent
from agents.tools_agent import ToolsAgent
from config import init_components, MEMORY_CONFIG
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)

# Define workflow events
class QueryEvent(Event):
    query: str
    session_id: str

class KeywordsEvent(Event):
    keywords: List[str]
    query: str
    session_id: str

class ChatHistoryEvent(Event):
    chat_history: List[ChatMessage]
    query: str
    session_id: str
    keywords: List[str]

class AnswerEvent(Event):
    answer: str
    query: str
    session_id: str
    chat_history: List[ChatMessage]
    keywords: List[str]

class QueryWorkflow(Workflow):
    """
    Workflow implementation for handling user queries.
    """
    def __init__(self):
        super().__init__(timeout=60, verbose=False)
        llm, embed_model, vector_store, _ = init_components()  # Ignore rate_limiter since it's a global instance
        self.memory_agent = MemoryAgent(
            llm=llm,
            vector_store=vector_store,
            embed_model=embed_model,
            **MEMORY_CONFIG
        )
        self.rag_agent = RAGAgent()
        self.tools_agent = ToolsAgent()
        self.logger = logging.getLogger(__name__)

    @step
    async def extract_keywords(self, ev: StartEvent) -> KeywordsEvent:
        """Extract keywords from the query."""
        query = ev.query
        session_id = ev.session_id if ev.session_id else str(uuid.uuid4())
        
        self.logger.info(f"Processing query in session {session_id}: {query}")
        keywords_result = self.tools_agent.use_tool('extract_keywords', query)
        
        # Convert keyword dictionaries to strings
        keywords = []
        if isinstance(keywords_result, list):
            keywords = [item['keyword'] for item in keywords_result if isinstance(item, dict) and 'keyword' in item]
        
        return KeywordsEvent(
            keywords=keywords,
            query=query,
            session_id=session_id
        )

    @step
    async def get_chat_history(self, ev: KeywordsEvent) -> ChatHistoryEvent:
        """Retrieve chat history from memory."""
        chat_history = await self.memory_agent.get_chat_history(ev.session_id)
        
        return ChatHistoryEvent(
            chat_history=chat_history,
            query=ev.query,
            session_id=ev.session_id,
            keywords=ev.keywords
        )

    @step
    async def generate_answer(self, ev: ChatHistoryEvent) -> AnswerEvent:
        """Generate answer using RAG agent."""
        answer = self.rag_agent.answer(ev.query, {'chat_history': ev.chat_history})
        
        return AnswerEvent(
            answer=answer,
            query=ev.query,
            session_id=ev.session_id,
            chat_history=ev.chat_history,
            keywords=ev.keywords
        )

    @step
    async def store_interaction(self, ev: AnswerEvent) -> StopEvent:
        """Store the interaction in memory and return final result."""
        messages = [
            ChatMessage(role="user", content=ev.query),
            ChatMessage(role="assistant", content=ev.answer)
        ]
        await self.memory_agent.add_messages(ev.session_id, messages)
        
        return StopEvent(result={
            'session_id': ev.session_id,
            'answer': ev.answer,
            'context_used': bool(ev.chat_history),
            'keywords': ev.keywords
        })

class CoreAgent:
    """
    Orchestrates the workflow, routes queries, manages memory and tool agents.
    """
    def __init__(self):
        self.workflow = QueryWorkflow()
        self.logger = logging.getLogger(__name__)
        # Initialize tools agent for document analysis
        self.tools_agent = ToolsAgent()
        
        # Initialize RAG agent with documents
        try:
            from llama_index.core import SimpleDirectoryReader, download_loader
            import os
            
            # Get PDF loader
            PDFReader = download_loader("PDFReader")
            pdf_reader = PDFReader()
            
            docs_dir = os.getenv('DOCUMENT_STORE_DIR', './data/documents')
            documents = []
            
            if os.path.exists(docs_dir):
                # Try to load each document individually
                for file in os.listdir(docs_dir):
                    if file.lower().endswith('.pdf'):
                        try:
                            file_path = os.path.join(docs_dir, file)
                            pdf_docs = pdf_reader.load_data(file=file_path)
                            documents.extend(pdf_docs)
                            self.logger.info(f"Successfully loaded {file}")
                        except Exception as e:
                            self.logger.error(f"Failed to load file {file} with error: {str(e)}. Skipping...")
                            continue
                    else:
                        try:
                            file_path = os.path.join(docs_dir, file)
                            text_docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
                            documents.extend(text_docs)
                            self.logger.info(f"Successfully loaded {file}")
                        except Exception as e:
                            self.logger.error(f"Failed to load file {file} with error: {str(e)}. Skipping...")
                            continue
                
                if documents:
                    self.workflow.rag_agent.initialize_retrievers(documents)
                    self.logger.info(f"Initialized RAG agent with {len(documents)} documents from {docs_dir}")
                else:
                    self.logger.warning("No documents were successfully loaded. RAG capabilities will be limited.")
            else:
                self.logger.warning(f"Documents directory {docs_dir} not found. RAG capabilities will be limited.")
        except Exception as e:
            self.logger.error(f"Error initializing RAG agent with documents: {str(e)}")

    async def handle_query(self, query: str, session_id: Optional[str] = None) -> Dict:
        """Main entry point for handling user queries."""
        try:
            handler = self.workflow.run(query=query, session_id=session_id)
            workflow_result = await handler
            
            # Ensure we have a proper result dictionary
            if not workflow_result or not hasattr(workflow_result, 'result'):
                # If workflow fails, try to get a direct response from RAG agent
                answer = self.workflow.rag_agent.get_fallback_response(query)
                return {
                    'session_id': session_id or str(uuid.uuid4()),
                    'answer': answer,
                    'keywords': [],
                    'context_used': False
                }
                
            return {
                'session_id': workflow_result.result.get('session_id', session_id or str(uuid.uuid4())),
                'answer': workflow_result.result.get('answer', ''),
                'keywords': workflow_result.result.get('keywords', []),
                'context_used': workflow_result.result.get('context_used', False)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            # Try to get a fallback response
            try:
                answer = self.workflow.rag_agent.get_fallback_response(query)
            except:
                answer = "I encountered an error processing your query. Please try again later."
                
            return {
                'session_id': session_id or str(uuid.uuid4()),
                'error': str(e),
                'answer': answer,
                'keywords': [],
                'context_used': False
            }

    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Analyze a document using available tools."""
        try:
            structure = self.tools_agent.use_tool('analyze_structure', text)
            themes = self.tools_agent.use_tool('analyze_themes', text)
            summary = self.tools_agent.use_tool('summarize', text)
            keywords = self.tools_agent.use_tool('extract_keywords', text)
            
            return {
                'structure': structure,
                'themes': themes,
                'summary': summary,
                'keywords': keywords
            }
        except Exception as e:
            self.logger.error(f"Error analyzing document: {str(e)}")
            return {'error': str(e)} 
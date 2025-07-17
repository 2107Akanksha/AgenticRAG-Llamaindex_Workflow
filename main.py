import logging
from agents.core_agent import CoreAgent
from llama_index.core.llms import ChatMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_conversation(session_id: str = "test_session"):
    """Process a sample conversation using the core agent."""
    # Initialize core agent
    core_agent = CoreAgent()
    
    # Process some sample queries
    queries = [
        # Basic Interaction Tests
        "Hello! Can you introduce yourself?",
        "What are your main capabilities?",
        
        # Knowledge-Based Questions (Testing RAG capabilities)
        "Can you summarize the key points from the Adobe annual report?",
        "What were Adobe's main financial highlights from the report?",
        "Compare Adobe's performance across different business segments.",
        
        # Memory and Context Testing
        "Let's talk about artificial intelligence.",
        "What did we just discuss?",  # Testing if it remembers the AI discussion
        "Going back to our earlier topic about Adobe, what were the key points?",  # Testing long-term memory
        
        # Tool Usage Tests
        "Can you extract the main keywords from this text: 'Adobe Creative Cloud saw significant growth in 2023, with digital media revenue reaching new heights.'",
        "Please analyze the sentiment of this statement: 'We are extremely pleased with our outstanding financial results.'",
        
        # Complex Queries (Testing reasoning and integration)
        "Based on the Adobe report, what are the company's main strengths and potential risks?",
        "Can you compare Adobe's current strategy with their previous year's approach?",
        
        # Error Handling and Edge Cases
        "Can you analyze a report that doesn't exist?",  # Testing error handling
        "",  # Testing empty input
        "This is a very very very very very very very very very very very very very very very long query that goes on and on and might exceed typical length limits",  # Testing long input
        
        # Multi-turn Conversation Tests
        "Let's analyze Adobe's market position.",
        "How does this compare to their competitors?",  # Context-dependent query
        "What factors contributed to this?",  # Testing follow-up understanding
    ]
    
    for query in queries:
        logger.info(f"Processing query: {query}")
        result = await core_agent.handle_query(query, session_id)
        logger.info(f"Response: {result['answer']}")
        logger.info(f"Keywords: {result['keywords']}")
        logger.info(f"Context used: {result['context_used']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(process_conversation())

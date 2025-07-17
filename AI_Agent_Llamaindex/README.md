# AI Agent with RAG and Memory Management

An intelligent AI agent system that combines Retrieval-Augmented Generation (RAG), persistent memory management, and specialized tools for enhanced document processing and analysis.

## Features

- **RAG (Retrieval-Augmented Generation)**
  - Document ingestion and processing
  - Semantic search capabilities
  - Context-aware responses
  - Built-in vector store using LlamaIndex
  - Advanced query planning with multi-hop reasoning
  - Query optimization based on interaction history
  - Intelligent query caching system

- **Memory Management**
  - Persistent memory storage using vector database
  - Session-based memory tracking
  - Long-term knowledge retention
  - Configurable memory storage options

- **Specialized Tools**
  - Content Analysis
  - Keyword Extraction
  - Document Summarization
  - Custom tool integration framework

## Getting Started

### Prerequisites
- Python 3.11 (preferable)

### Installation
```bash
cd AI_Agent_assignment
```
Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
# Required
GOOGLE_API_KEY=your_google_api_key

# Optional
VECTOR_STORE_DIR=path/to/vector/store
MEMORY_STORE_DIR=path/to/memory/store
DOCUMENT_STORE_DIR=path/to/document/store
LOG_LEVEL=INFO
LOG_FILE=path/to/log/file
```

## Usage

### Basic Usage

```python
from agents.core_agent import CoreAgent

# Initialize the agent
agent = CoreAgent()

# Process a document
agent.process_document("path/to/document.pdf")

# Query the agent
response = agent.query("What insights can you provide from the document?")
```

### Advanced Query Planning

The system now includes sophisticated query planning capabilities:

```python
from agents.rag_agent import RAGAgent

# Initialize RAG agent
rag_agent = RAGAgent()

# Multi-hop reasoning example
response = rag_agent.answer(
    "Compare the efficiency and cost of solar vs wind energy, "
    "considering both initial setup and long-term maintenance"
)

# The query planner will automatically:
# 1. Break down the complex query into steps
# 2. Execute each step with proper dependencies
# 3. Synthesize the final answer
```

### Query Caching and Optimization

The system includes intelligent caching and query optimization:

```python
# Cached responses are automatically used for identical queries
response1 = rag_agent.answer("What is the cost of solar panels?")
response2 = rag_agent.answer("What is the cost of solar panels?")  # Uses cache

# Query optimization based on context
response = rag_agent.answer(
    "How does it compare to wind?",
    context={'previous_query': 'What is the cost of solar panels?'}
)
```

### Memory Management

```python
from agents.memory_agent import MemoryAgent

# Initialize memory agent
memory_agent = MemoryAgent()

# Store information
memory_agent.store_memory("Important information to remember")

# Retrieve relevant memories
memories = memory_agent.retrieve_memories("query context")
```

### Using Tools

```python
from tools.content_analyzer import ContentAnalyzer
from tools.summarizer import Summarizer

# Analyze content
analyzer = ContentAnalyzer()
analysis = analyzer.analyze("text to analyze")

# Generate summary
summarizer = Summarizer()
summary = summarizer.summarize("text to summarize")
```

## Configuration

The system can be configured through various settings in `config.py`:

- Vector store settings
- Memory persistence options
- Tool configurations
- Logging preferences
- Query cache settings
- Query planning parameters

### Query Planning Configuration

You can customize the query planning behavior:

```python
from agents.query_planner import QueryPlanner

planner = QueryPlanner(
    cache_dir="custom/cache/dir",  # Custom cache directory
)

# Configure RAG agent with custom planner
rag_agent = RAGAgent()
rag_agent.query_planner = planner
```

## Advanced Features

### Multi-hop Reasoning

The system supports complex multi-hop reasoning:

1. **Query Decomposition**: Breaks down complex queries into interconnected steps
2. **Dependency Tracking**: Manages information flow between reasoning steps
3. **Context Preservation**: Maintains context across multiple hops
4. **Answer Synthesis**: Combines intermediate results into coherent responses

### Query Optimization

Queries are automatically optimized based on:

1. **Historical Interactions**: Learning from past query patterns
2. **Context Awareness**: Considering current conversation context
3. **Caching Strategy**: Intelligent caching of frequent queries
4. **Resource Management**: Optimizing resource usage for complex queries

### Caching System

The caching system provides:

1. **Intelligent Caching**: Automatically caches frequent and complex queries
2. **Cache Invalidation**: Time-based cache invalidation (24-hour default)
3. **Memory Efficiency**: LRU-based cache management
4. **Persistence**: Disk-based cache storage for reliability

## Documentation
Each module includes detailed documentation:
- `agents/`: Core agent implementations and behaviors
- `memory/`: Memory management and persistence
- `tools/`: Available tools and their usage
- `retriever/`: Document retrieval and processing


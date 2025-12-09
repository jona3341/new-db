# GraphRAG for MySQL Database

This project implements GraphRAG (Graph Retrieval-Augmented Generation) for your MySQL database. It combines graph database structure with RAG for intelligent querying and understanding of your database schema.

## Features

- **Automatic Schema Extraction**: Extracts table structures, columns, and relationships from your MySQL database
- **Knowledge Graph Building**: Creates a graph representation of your database schema
- **RAG Integration**: Uses LangChain and DeepSeek v3 for intelligent question answering (no OpenAI required)
- **Vector Search**: Uses ChromaDB for semantic search over database schema
- **100% OpenAI-Free**: Uses sentence-transformers for embeddings and direct HTTP requests to DeepSeek API

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. API Configuration

The DeepSeek API key is already configured in `config.py`. 

**Note**: This system is **completely OpenAI-free**:
- ✅ Uses **sentence-transformers** for embeddings (local, no API needed)
- ✅ Uses **direct HTTP requests** to DeepSeek API (no OpenAI SDK)
- ✅ Works offline for embeddings after initial model download
- ✅ No location restrictions
- ✅ Free embeddings, only DeepSeek API calls are charged

The embedding model will be downloaded automatically on first use (about 80MB).

### 3. Database Configuration

The database configuration is already set in `config.py`. You can modify it if needed:

```python
DB_CONFIG = {
    'host': '10.46.0.7',
    'port': 9030,
    'user': 'test_user',
    'password': 'test@123456',
    'db': 'zjyy_bigdata',
    'charset': 'utf8mb4'
}
```

## Usage

### Basic Usage

```python
from graphrag import GraphRAG
from config import DB_CONFIG, GRAPH_CONFIG

# Initialize GraphRAG
graphrag = GraphRAG(DB_CONFIG, GRAPH_CONFIG)
graphrag.initialize()

# Query the system
result = graphrag.query("What tables are in this database?")
print(result['answer'])

# Close connections
graphrag.close()
```

### Running the Example

```bash
python graphrag.py
```

## How It Works

1. **Schema Extraction**: Connects to your MySQL database and extracts:
   - All tables and their columns
   - Data types and primary keys
   - Foreign key relationships

2. **Graph Building**: Creates a NetworkX graph where:
   - Nodes represent tables and columns
   - Edges represent relationships (has_column, foreign_key)

3. **Text Generation**: Converts the graph structure into a text summary

4. **Vectorization**: Splits the text into chunks and creates embeddings using OpenAI

5. **RAG Querying**: When you ask a question:
   - The system searches the vector store for relevant schema information
   - Uses the retrieved context with an LLM to generate an answer

## Configuration

You can customize the behavior in `config.py`:

- `GRAPH_CONFIG`: Control graph size limits and chunking parameters
- `OPENAI_CONFIG`: Adjust the LLM model and temperature

## Example Queries

- "What tables are in this database?"
- "What are the relationships between tables?"
- "Show me the schema of table X"
- "What columns does table Y have?"
- "How are table A and table B related?"

## Requirements

- Python 3.8+
- MySQL database access
- OpenAI API key
- Network connectivity to database (10.46.0.7:9030)

## Notes

- The system only extracts schema information, not actual data (for privacy and performance)
- Make sure your database user has permissions to query INFORMATION_SCHEMA
- The vector store is created locally in a `chroma_db` directory


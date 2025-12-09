"""
Database configuration for GraphRAG
"""
DB_CONFIG = {
    'host': '10.46.0.7',
    'port': 9030,
    'user': 'test_user',
    'password': 'test@123456',
    'db': 'zjyy_bigdata',
    'charset': 'utf8mb4'
}

# GraphRAG Configuration
GRAPH_CONFIG = {
    'max_nodes': 10000,  # Maximum number of nodes in the graph
    'max_edges': 50000,  # Maximum number of edges in the graph
    'chunk_size': 1000,  # Text chunk size for embeddings
    'chunk_overlap': 200,  # Overlap between chunks
}

# DeepSeek Configuration
DEEPSEEK_CONFIG = {
    'api_key': 'sk-485fa635a9744e4a8e921d3a65a66a91',
    'model': 'deepseek-chat',  # DeepSeek v3 model (or 'deepseek-reasoner' for reasoning)
    'temperature': 0.7,
    'base_url': 'https://api.deepseek.com/v1',  # DeepSeek API endpoint
}

# Embedding Configuration (using sentence-transformers - works locally, no API needed)
EMBEDDING_CONFIG = {
    'provider': 'sentence-transformers',  # Local embeddings, no API required
    'model': 'all-MiniLM-L6-v2',  # Fast and efficient multilingual model
    # Alternative models you can use:
    # 'all-mpnet-base-v2' - Better quality, slower
    # 'paraphrase-multilingual-MiniLM-L12-v2' - Multilingual support
}


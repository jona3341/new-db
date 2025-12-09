"""
Simplified GraphRAG implementation for MySQL database
Uses minimal dependencies - only what's needed for DeepSeek and RAG
"""
import pymysql
import networkx as nx
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests
import os
from dotenv import load_dotenv
from config import DB_CONFIG, GRAPH_CONFIG, DEEPSEEK_CONFIG, EMBEDDING_CONFIG

load_dotenv()


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Simple text splitter"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
        if start >= len(text):
            break
    return chunks


class DeepSeekClient:
    """Simple DeepSeek API client"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat", temperature: float = 0.7, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send chat request to DeepSeek API"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}/chat/completions"
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error calling DeepSeek API: {e}")


class DatabaseGraphBuilder:
    """Builds a knowledge graph from MySQL database schema and data"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.connection = None
        self.graph = nx.DiGraph()
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = pymysql.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['db'],
                charset=self.db_config['charset'],
                cursorclass=pymysql.cursors.DictCursor
            )
            print(f"Connected to database: {self.db_config['db']}")
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def get_tables(self) -> List[str]:
        """Get all table names from the database"""
        with self.connection.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = [list(row.values())[0] for row in cursor.fetchall()]
        return tables
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a table"""
        with self.connection.cursor() as cursor:
            cursor.execute(f"DESCRIBE {table_name}")
            return cursor.fetchall()
    
    def get_foreign_keys(self, table_name: str) -> List[Dict[str, Any]]:
        """Extract foreign key relationships"""
        with self.connection.cursor() as cursor:
            query = """
            SELECT 
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = %s
            AND TABLE_NAME = %s
            AND REFERENCED_TABLE_NAME IS NOT NULL
            """
            cursor.execute(query, (self.db_config['db'], table_name))
            return cursor.fetchall()
    
    def build_schema_graph(self):
        """Build graph from database schema"""
        if not self.connection:
            self.connect()
        
        tables = self.get_tables()
        print(f"Found {len(tables)} tables")
        
        # Add table nodes
        for table in tables:
            self.graph.add_node(f"table:{table}", type="table", name=table)
            
            # Get schema for each table
            schema = self.get_table_schema(table)
            for column in schema:
                col_name = column['Field']
                col_type = column['Type']
                is_primary = column['Key'] == 'PRI'
                
                node_id = f"column:{table}.{col_name}"
                self.graph.add_node(
                    node_id,
                    type="column",
                    table=table,
                    name=col_name,
                    data_type=col_type,
                    is_primary=is_primary
                )
                # Link column to table
                self.graph.add_edge(f"table:{table}", node_id, relation="has_column")
            
            # Add foreign key relationships
            foreign_keys = self.get_foreign_keys(table)
            for fk in foreign_keys:
                source_table = table
                target_table = fk['REFERENCED_TABLE_NAME']
                source_col = fk['COLUMN_NAME']
                target_col = fk['REFERENCED_COLUMN_NAME']
                
                # Add edge between tables
                self.graph.add_edge(
                    f"table:{source_table}",
                    f"table:{target_table}",
                    relation="foreign_key",
                    source_column=source_col,
                    target_column=target_col
                )
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def get_graph_summary(self) -> str:
        """Generate a text summary of the graph structure"""
        summary_parts = []
        
        # Get all tables
        tables = [n for n in self.graph.nodes() if self.graph.nodes[n].get('type') == 'table']
        
        summary_parts.append(f"Database: {self.db_config['db']}")
        summary_parts.append(f"Total Tables: {len(tables)}")
        summary_parts.append("\nTables and their columns:")
        
        for table_node in tables:
            table_name = self.graph.nodes[table_node]['name']
            summary_parts.append(f"\nTable: {table_name}")
            
            # Get columns for this table
            columns = [n for n in self.graph.neighbors(table_node) 
                      if self.graph.nodes[n].get('type') == 'column']
            
            for col_node in columns:
                col_info = self.graph.nodes[col_node]
                col_type = col_info.get('data_type', 'unknown')
                is_primary = col_info.get('is_primary', False)
                pk_marker = " (PRIMARY KEY)" if is_primary else ""
                summary_parts.append(f"  - {col_info['name']}: {col_type}{pk_marker}")
            
            # Get foreign key relationships
            fk_edges = [(u, v, d) for u, v, d in self.graph.edges(data=True)
                       if u == table_node and d.get('relation') == 'foreign_key']
            
            for source, target, data in fk_edges:
                target_table = self.graph.nodes[target]['name']
                summary_parts.append(f"  -> Foreign key to: {target_table}")
        
        return "\n".join(summary_parts)
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("Database connection closed")


class GraphRAG:
    """Simplified GraphRAG implementation using only DeepSeek and essential libraries"""
    
    def __init__(self, db_config: Dict[str, Any], graph_config: Dict[str, Any]):
        self.db_config = db_config
        self.graph_config = graph_config
        self.graph_builder = DatabaseGraphBuilder(db_config)
        self.vectorstore = None
        self.embeddings = None
        self.deepseek_client = None
        
    def initialize(self):
        """Initialize GraphRAG system"""
        print("Initializing GraphRAG...")
        
        # Build graph from database schema
        self.graph_builder.connect()
        self.graph_builder.build_schema_graph()
        
        # Generate graph summary
        graph_summary = self.graph_builder.get_graph_summary()
        
        # Initialize embeddings using sentence-transformers directly
        print(f"Loading embedding model: {EMBEDDING_CONFIG['model']}...")
        self.embeddings = SentenceTransformer(EMBEDDING_CONFIG['model'])
        print("Embedding model loaded successfully!")
        
        # Split text into chunks
        texts = split_text(
            graph_summary,
            chunk_size=self.graph_config['chunk_size'],
            chunk_overlap=self.graph_config['chunk_overlap']
        )
        
        # Create embeddings for all chunks
        embeddings_list = self.embeddings.encode(texts, normalize_embeddings=True)
        
        # Create ChromaDB vector store
        client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        # Get or create collection
        try:
            collection = client.get_collection("db_schema")
            collection.delete()  # Clear existing
        except:
            pass
        
        collection = client.create_collection(
            name="db_schema",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add documents and embeddings
        ids = [f"chunk_{i}" for i in range(len(texts))]
        collection.add(
            ids=ids,
            embeddings=embeddings_list.tolist(),
            documents=texts
        )
        
        self.vectorstore = collection
        
        # Initialize DeepSeek client
        deepseek_api_key = DEEPSEEK_CONFIG.get('api_key') or os.getenv('DEEPSEEK_API_KEY')
        if not deepseek_api_key:
            raise ValueError(
                "DeepSeek API key not found. "
                "Set it in config.py or as DEEPSEEK_API_KEY environment variable."
            )
        
        self.deepseek_client = DeepSeekClient(
            api_key=deepseek_api_key,
            model=DEEPSEEK_CONFIG['model'],
            temperature=DEEPSEEK_CONFIG['temperature'],
            base_url=DEEPSEEK_CONFIG.get('base_url', 'https://api.deepseek.com/v1')
        )
        
        print("GraphRAG initialized successfully!")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the GraphRAG system"""
        if not self.deepseek_client or not self.vectorstore:
            raise ValueError("GraphRAG not initialized. Call initialize() first.")
        
        # Get query embedding
        query_embedding = self.embeddings.encode([question], normalize_embeddings=True)[0]
        
        # Search for relevant chunks
        results = self.vectorstore.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )
        
        # Combine context from retrieved documents
        context = "\n\n".join(results['documents'][0])
        
        # Create prompt
        prompt = f"""You are a database expert assistant. Use the following database schema information to answer questions.

Database Schema Context:
{context}

Question: {question}

Provide a detailed answer based on the database schema. If you need to write SQL queries, provide them in a clear format.
Answer:"""
        
        # Call DeepSeek API
        messages = [{"role": "user", "content": prompt}]
        answer = self.deepseek_client.chat(messages)
        
        return {
            "answer": answer,
            "source_documents": results['documents'][0] if results['documents'] else []
        }
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the graph"""
        graph = self.graph_builder.graph
        return {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "tables": len([n for n in graph.nodes() if graph.nodes[n].get('type') == 'table'])
        }
    
    def close(self):
        """Clean up resources"""
        self.graph_builder.close()


def main():
    """Example usage"""
    # Initialize GraphRAG
    graphrag = GraphRAG(DB_CONFIG, GRAPH_CONFIG)
    
    try:
        graphrag.initialize()
        
        # Get graph info
        info = graphrag.get_graph_info()
        print(f"\nGraph Info: {info}")
        
        # Example queries
        questions = [
            "What tables are in this database?",
            "What are the relationships between tables?",
            "Show me the schema of the main tables"
        ]
        
        for question in questions:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"{'='*60}")
            result = graphrag.query(question)
            print(f"Answer: {result['answer']}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        graphrag.close()


if __name__ == "__main__":
    main()


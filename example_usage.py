"""
Example usage of GraphRAG
"""
from graphrag import GraphRAG
from config import DB_CONFIG, GRAPH_CONFIG

def main():
    """Example usage of GraphRAG"""
    # Initialize GraphRAG
    print("Creating GraphRAG instance...")
    graphrag = GraphRAG(DB_CONFIG, GRAPH_CONFIG)
    
    try:
        # Initialize the system (this will connect to DB and build the graph)
        graphrag.initialize()
        
        # Get graph information
        info = graphrag.get_graph_info()
        print(f"\n{'='*60}")
        print(f"Graph Information:")
        print(f"  - Nodes: {info['nodes']}")
        print(f"  - Edges: {info['edges']}")
        print(f"  - Tables: {info['tables']}")
        print(f"{'='*60}\n")
        
        # Example queries
        example_questions = [
            "What tables are in this database?",
            "What are the relationships between tables?",
            "Describe the database schema",
        ]
        
        print("You can now ask questions about your database schema.")
        print("Example questions:")
        for i, q in enumerate(example_questions, 1):
            print(f"  {i}. {q}")
        print("\n" + "="*60)
        
        # Interactive mode
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            try:
                result = graphrag.query(question)
                print(f"\nAnswer:\n{result['answer']}")
                
                if result.get('source_documents'):
                    print(f"\n(Retrieved {len(result['source_documents'])} relevant documents)")
            except Exception as e:
                print(f"Error: {e}")
        
    except Exception as e:
        print(f"Error initializing GraphRAG: {e}")
        print("\nMake sure:")
        print("1. Your database is accessible at 10.46.0.7:9030")
        print("2. You have set OPENAI_API_KEY in your .env file")
        print("3. You have installed all requirements: pip install -r requirements.txt")
    finally:
        graphrag.close()
        print("\nGraphRAG session ended.")


if __name__ == "__main__":
    main()


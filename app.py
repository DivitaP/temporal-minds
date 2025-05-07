from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os

# Import our RAG components
from src.temporal_minds.pipeline.temporal_retriver import Neo4jConnector, KnowledgeEmbedder
from src.temporal_minds.pipeline.temporal_generator import LLMGenerator, TuringKnowledgeGraph

# Configuration
NEO4J_URI = "bolt://127.0.0.1:7687"  
NEO4J_USER = "neo4j"                 
NEO4J_PASSWORD = "password"         

# Time scope of the knowledge graph
TIMELINE_START = "1912"  
TIMELINE_END = "1939"    

# Initialize Flask app
app = Flask(__name__)
# Allow frontend origin
CORS(app, resources={r"/chat": {"origins": "http://localhost:3000"}})

# Initialize the Turing Knowledge Graph RAG system
try:
    # Create the components
    neo4j_connector = Neo4jConnector(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
    
    knowledge_embedder = KnowledgeEmbedder()
    
    llm_generator = LLMGenerator(model_choice="flan")
    
    # Connect everything in the main RAG system
    kg_rag = TuringKnowledgeGraph(
        neo4j_connector=neo4j_connector,
        knowledge_embedder=knowledge_embedder,
        generator=llm_generator,
        timeline_scope=(TIMELINE_START, TIMELINE_END)
    )
    
    print("Turing Knowledge Graph RAG system initialized successfully!")
except Exception as e:
    print(f"Error initializing RAG system: {e}")
    # Fallback function in case Neo4j is not available
    def simple_turing_response(message):
        return (f"As Alan Turing, I would like to discuss '{message}', but my knowledge graph "
                f"database is currently unavailable. Please ensure Neo4j is running correctly.")
    kg_rag = None

@cross_origin()
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    
    # Process query using the RAG system if available
    if kg_rag:
        try:
            bot_reply = kg_rag.process_query(user_message)
        except Exception as e:
            print(f"Error processing query: {e}")
            bot_reply = f"I encountered an issue processing your question. Please try again."
    else:
        bot_reply = simple_turing_response(user_message)
    
    return jsonify({"response": bot_reply})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
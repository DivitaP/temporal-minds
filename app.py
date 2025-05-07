from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Import your RAG components
from neo4j import GraphDatabase

# Configuration
NEO4J_URI = "bolt://127.0.0.1:7687"  
NEO4J_USER = "neo4j"                 
NEO4J_PASSWORD = "password"         

# Time scope of the knowledge graph
TIMELINE_START = "1912"  
TIMELINE_END = "1939"    

class Neo4jConnector:
    """Connects to Neo4j database and retrieves data about Alan Turing."""
    
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()

    def get_timeline_events(self):
        """Get all events with dates from the knowledge graph."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:EVENT)
                WHERE n.date IS NOT NULL
                RETURN n.label AS event, n.date AS date, n.description AS description
                UNION
                MATCH (n)-[r]->(m)
                WHERE r.date IS NOT NULL
                RETURN r.description AS event, r.date AS date, r.description AS description
            """)
            
            events = [{"event": record["event"], 
                    "date": record["date"],
                    "description": record["description"]} for record in result]
            return events
        
    def get_all_knowledge(self):
        """Extract all knowledge statements from the graph."""
        with self.driver.session() as session:
            # Query to extract relationships and properties as natural language
            result = session.run("""
                MATCH (n)-[r]->(m)
                WHERE r.description IS NOT NULL
                RETURN r.description AS statement
                UNION
                MATCH (n)
                WHERE n.description IS NOT NULL
                RETURN n.description AS statement
            """)
            statements = [record["statement"] for record in result]
            print(f"Retrieved {len(statements)} statements from Neo4j.")
            return statements
            
    def query_knowledge(self, query_terms):
        """
        Query the knowledge graph with specific terms, providing more comprehensive
        and flexible search capabilities.
        """
        # Normalize and process query terms
        query_terms = query_terms.lower().strip()
        
        with self.driver.session() as session:
            statements = []
            
            # Search for relationships where the description, source entity, or target entity 
            # matches the query terms
            result = session.run("""
                MATCH (n)-[r]->(m)
                WHERE r.description IS NOT NULL AND 
                    (toLower(n.label) CONTAINS $query_param OR 
                    toLower(m.label) CONTAINS $query_param OR 
                    toLower(r.description) CONTAINS $query_param)
                RETURN DISTINCT r.description AS statement, 
                    n.label AS source, 
                    type(r) AS relationship,
                    m.label AS target
                LIMIT 25
            """, query_param=query_terms)
            
            for record in result:
                statements.append({
                    "statement": record["statement"],
                    "context": f"{record['source']} {record['relationship']} {record['target']}"
                })
            
            # Search for entity descriptions that match the query terms
            result = session.run("""
                MATCH (n)
                WHERE n.description IS NOT NULL AND 
                    (toLower(n.label) CONTAINS $query_param OR 
                    toLower(n.description) CONTAINS $query_param)
                RETURN n.label AS entity, 
                    n.description AS statement,
                    labels(n)[0] AS entity_type
                LIMIT 25
            """, query_param=query_terms)
            
            for record in result:
                statements.append({
                    "statement": record["statement"],
                    "context": f"{record['entity_type']}: {record['entity']}"
                })
            
            # If no results found, try breaking the query into tokens and matching any of them
            if not statements:
                query_tokens = query_terms.split()
                if len(query_tokens) > 1:
                    query_conditions = " OR ".join([f"toLower(n.label) CONTAINS '{token}' OR toLower(m.label) CONTAINS '{token}' OR toLower(r.description) CONTAINS '{token}'" for token in query_tokens])
                    
                    result = session.run(f"""
                        MATCH (n)-[r]->(m)
                        WHERE r.description IS NOT NULL AND ({query_conditions})
                        RETURN DISTINCT r.description AS statement, 
                            n.label AS source, 
                            type(r) AS relationship,
                            m.label AS target
                        LIMIT 25
                    """)
                    
                    for record in result:
                        statements.append({
                            "statement": record["statement"],
                            "context": f"{record['source']} {record['relationship']} {record['target']}"
                        })
            
            # Implement specific queries for known entity types if standard search returns limited results
            if len(statements) < 5 and any(term in query_terms for term in ['person', 'turing']):
                # Specific query for persons and their relationships
                result = session.run("""
                    MATCH (p:PERSON)-[r]->(m)
                    WHERE toLower(p.label) CONTAINS 'turing'
                    RETURN DISTINCT r.description AS statement, 
                        p.label AS source, 
                        type(r) AS relationship,
                        m.label AS target
                    LIMIT 15
                """)
                
                for record in result:
                    statements.append({
                        "statement": record["statement"],
                        "context": f"{record['source']} {record['relationship']} {record['target']}"
                    })
                    
            return statements

class KnowledgeEmbedder:
    """Creates and searches vector embeddings for knowledge graph statements."""
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.statements = []
        
    def build_index(self, statements):
        """Build a FAISS index from knowledge statements."""
        self.statements = statements
        embeddings = self.embedder.encode(statements, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        return self
        
    def search(self, query, top_k=5):
        """Search for most relevant statements to the query."""
        if not self.index:
            raise ValueError("Index not built. Call build_index first.")
            
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "statement": self.statements[idx],
                "relevance": float(1 / (1 + distances[0][i]))  # Convert distance to relevance score
            })
        
        return results

class LLMGenerator:
    """Uses an LLM to generate answers based on retrieved knowledge."""
    
    def __init__(self, model_choice="flan"):
        self.generator = self._load_generator(model_choice)
        
    def _load_generator(self, model_choice):
        device = -1  # CPU, use 0 or specific GPU ID if available
        
        if model_choice == "flan":
            return pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                device=device
            )
        else:
            raise NotImplementedError(f"Model {model_choice} not implemented")
    
    def generate_answer(self, query, retrieved_facts, timeline_scope=None):
        """Generate an answer based on retrieved facts."""
        if not retrieved_facts:
            return self._generate_no_information_response(query, timeline_scope)
        
        # Combine retrieved facts
        context = "\n".join(retrieved_facts)
        
        if timeline_scope:
            time_constraint = f"Only use information from the time period {timeline_scope[0]} to {timeline_scope[1]}."
        else:
            time_constraint = ""
        
        prompt = f"""
        You are an AI assistant knowledgeable about Alan Turing's life and work.
        
        Based only on the following retrieved facts:
        {context}
        
        {time_constraint}
        
        Answer the user's question:
        {query}
        
        If the information to answer the question is not contained in the facts above,
        say that you don't have enough information in your knowledge base and explain
        the time period your knowledge covers (1911-1938).
        """
        
        result = self.generator(prompt, max_length=256, do_sample=True, temperature=0.5)
        return result[0]['generated_text']
    
    def _generate_no_information_response(self, query, timeline_scope=None):
        """Generate a response when no information is available."""
        if timeline_scope:
            return f"I don't have information about that in my knowledge base. My knowledge about Alan Turing covers the period from {timeline_scope[0]} to {timeline_scope[1]}, focusing on his early life, education, and early academic work including his papers on computation and his dissertation."
        else:
            return "I don't have information about that in my knowledge base. My knowledge about Alan Turing is limited to specific events and relationships documented in the available data."

class TuringKnowledgeGraph:
    """Main class connecting Neo4j KG with the RAG pipeline."""
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, model_choice="flan"):
        self.neo4j = Neo4jConnector(neo4j_uri, neo4j_user, neo4j_password)
        self.embedder = KnowledgeEmbedder()
        self.generator = LLMGenerator(model_choice)
        self.timeline_scope = (TIMELINE_START, TIMELINE_END)
        self.all_knowledge = []
        
        # Initialize the system
        self._initialize()
        
    def _initialize(self):
        """Initialize the knowledge graph and build embeddings."""
        print("Initializing Turing Knowledge Graph RAG system...")
        print("Retrieving knowledge from Neo4j...")
        self.all_knowledge = self.neo4j.get_all_knowledge()
        
        print(f"Building embeddings for {len(self.all_knowledge)} knowledge statements...")
        self.embedder.build_index(self.all_knowledge)
        
        self.timeline_events = self.neo4j.get_timeline_events()
        print("System initialized successfully!")
        print(f"Knowledge scope: {self.timeline_scope[0]} - {self.timeline_scope[1]}")
    
    def is_query_in_timeline(self, query):
        """Check if a query mentions years outside our timeline scope."""
        # Extract years from the query
        year_pattern = r'\b(19\d{2}|20\d{2})\b'  # Match years from 1900-2099
        mentioned_years = re.findall(year_pattern, query)
        
        if not mentioned_years:
            return True  # No years mentioned, assume in timeline
            
        for year in mentioned_years:
            if int(year) < int(self.timeline_scope[0]) or int(year) > int(self.timeline_scope[1]):
                return False
                
        return True
        
    def process_query(self, query):
        """Process a user query and generate an answer."""
        if not self.is_query_in_timeline(query):
            return f"I don't have information about events outside my knowledge timeline, which covers {self.timeline_scope[0]}-{self.timeline_scope[1]}. This period includes Alan Turing's early life, education at Sherborne School and Cambridge, his work on computable numbers, and his PhD at Princeton."
        
        # First try direct Neo4j query for more structured data
        neo4j_results = self.neo4j.query_knowledge(query)
        
        if neo4j_results:
            # If we got direct matches from Neo4j, use those
            statements = [result["statement"] for result in neo4j_results]
            
            # Add relevant context from the statement contexts
            contexts = set(result.get("context", "") for result in neo4j_results)
            context_statement = "Relevant entities: " + ", ".join(contexts)
            statements.append(context_statement)
            
            return self.generator.generate_answer(query, statements, self.timeline_scope)
        
        # Fall back to semantic search if direct query didn't yield results
        search_results = self.embedder.search(query, top_k=5)
        
        if search_results:
            statements = [result["statement"] for result in search_results]
            return self.generator.generate_answer(query, statements, self.timeline_scope)
        
        # If both approaches failed, generate a "no information" response
        return self.generator._generate_no_information_response(query, self.timeline_scope)
    
    def close(self):
        """Clean up resources."""
        self.neo4j.close()

# Initialize Flask app
app = Flask(__name__)
# Allow frontend origin
CORS(app, resources={r"/chat": {"origins": "http://localhost:3000"}})

# Initialize the Turing Knowledge Graph RAG system
try:
    kg_rag = TuringKnowledgeGraph(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        model_choice="flan"
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
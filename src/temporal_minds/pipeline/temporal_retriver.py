from neo4j import GraphDatabase
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re

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
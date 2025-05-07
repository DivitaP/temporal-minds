from transformers import pipeline
import re

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
    
    def __init__(self, neo4j_connector, knowledge_embedder, generator, timeline_scope=("1912", "1939")):
        self.neo4j = neo4j_connector
        self.embedder = knowledge_embedder
        self.generator = generator
        self.timeline_scope = timeline_scope
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
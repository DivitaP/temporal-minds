# Alan Turing - Temporal Chatbot

This project report presents an approach to implement data unlearning in Language Models using a Neo4j knowledge graph with temporal constraints. We developed a Retrieval Augmented Generation (RAG) system that limits responses to information from a specific historical period (1912-1939), using Alan Turing’s life as our case study. Our results show that knowledge graphs provide an effective mechanism for temporal knowledge containment, though challenges remain in preventing model hallucination with ambiguous queries. This project contributes to research on controlled knowledge access in language models and provides a framework for creating historically accurate AI personas.

## Project Overview

This project consists of:

1. **Backend**: A Flask API that connects to a Neo4j knowledge graph database and uses RAG to generate responses
2. **Frontend**: A React application that provides a chat interface

The system leverages:
- Neo4j graph database for storing structured knowledge about Alan Turing
- Sentence transformers for semantic embedding of knowledge statements
- FAISS for efficient vector similarity search
- HuggingFace transformers for text generation
- Flask for backend API
- React for the frontend interface

## Prerequisites

- Python 3.8+
- Node.js 14+ and npm
- Neo4j Database (4.0+)

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:DivitaP/temporal-minds.git
cd temporal-minds
```

### 2. Set up Neo4j Database

1. [Download and install Neo4j](https://neo4j.com/download/) or use Neo4j Desktop
2. Create a new database with the following credentials:
   - URL: `bolt://127.0.0.1:7687`
   - Username: `neo4j`
   - Password: `password`
   
   (You can use different credentials, but make sure to update them in the backend code)

3. Import the Alan Turing knowledge graph data (note: You can find the cypher queries in root directory with cypher_texts.txt)

### 3. Set up Backend

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install flask flask-cors neo4j sentence-transformers faiss-cpu transformers torch
```

### 4. Set up Frontend

```bash
# Install dependencies
cd frontend
npm install
```

## Configuration

You can modify the backend configuration by editing the following variables in the Python file:

```python
# Configuration
NEO4J_URI = "bolt://127.0.0.1:7687"  
NEO4J_USER = "neo4j"                 
NEO4J_PASSWORD = "password"         

# Time scope of the knowledge graph
TIMELINE_START = "1912"  
TIMELINE_END = "1939"    
```

## Running the Application

### 1. Start Neo4j Database

Ensure your Neo4j database is running and accessible.

### 2. Start the Backend Server

```bash
# From the project root directory
python app.py
```

The Flask server should start on http://127.0.0.1:5000

### 3. Start the Frontend Development Server

```bash
# From the project root directory
cd frontend
npm start
```

The React app should open automatically at http://localhost:3000

## Using the Chatbot

1. Type your question about Alan Turing in the input field
2. Press Enter or click the Send button
3. The chatbot will respond with information based on the knowledge graph

The chatbot has knowledge about:
- Alan Turing's early life (1911-1939)
- His education at Sherborne School and Cambridge
- His work on computable numbers
- His PhD at Princeton
- Other academic work and personal life from this period

## Troubleshooting

### Connection Issues

If you see "Not connected to backend service" warning:
1. Ensure the Flask server is running
2. Check that it's running on port 5000
3. Verify CORS settings if you're hosting frontend and backend on different domains

### Neo4j Connection Issues

If there are errors connecting to Neo4j:
1. Verify Neo4j is running
2. Check your credentials in the configuration
3. Ensure the database contains the required knowledge graph data

## Project Structure

```
temporal-minds/
├── app.py                # Flask backend
├── cypher_texts.txt      # Cypher queries to create Alan Turing's Knowledge Graph
├── package.json          # Frontend dependencies
├── public/               # Frontend public assets
└── src/                  # Frontend React code
    ├── App.js            # Main React component  
    ├── App.css           # Styling
    └── index.js          # React entry point

```

## Future Improvements

1. Enhanced RAG Pipeline: Integrating more sophisticated language models
2. Multi-Persona Knowledge Graph: Expanding to include multiple historical figures
3. Improved Hallucination Prevention: Developing more robust verification mechanisms
4. Automated Knowledge Extraction: Creating pipelines for extracting temporally tagged information
5. Large Language Model Integration: Exploring how larger LLMs can be integrated while maintaining constraints



## Acknowledgements

- This project uses data about Alan Turing's life and contributions
- Built with React, Flask, Neo4j, and HuggingFace Transformers

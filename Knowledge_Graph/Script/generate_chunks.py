import json
import spacy

# Step 1: Load JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Sample data - Replace this with loading from the actual file
data = load_json("alan_turing.json")  # Uncomment this when using a real file

# Step 2: Load spaCy model for chunking and sentence segmentation
nlp = spacy.load("en_core_web_sm")

# Function to chunk text into 200-character sections based on sentence boundaries
def chunk_text(text, chunk_size=600):
    doc = nlp(text)  # Use spaCy to parse the text into sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence exceeds chunk_size, start a new chunk
        if len(current_chunk) + len(sentence) > chunk_size:
            if current_chunk:  # Ensure not appending empty chunks
                chunks.append(current_chunk.strip())
            current_chunk = sentence  # Start new chunk with the current sentence
        else:
            current_chunk += " " + sentence  # Add sentence to the current chunk
    
    if current_chunk:  # Add the last chunk if there's any remaining text
        chunks.append(current_chunk.strip())
    
    return chunks

# Step 3: Extract descriptions from JSON
def extract_descriptions(data):
    descriptions = []

    # Add the general description (this is from the 'description' key)
    general_desc = data["description"]
    print("General Description:", general_desc)  # Debugging line to check the general description
    
    general_desc_chunks = chunk_text(general_desc)  # Get chunks from the general description
    for chunk in general_desc_chunks:
        descriptions.append({
            "chunk": chunk,
            "original_event_name": "General Description",  # General description is not an event
            "time_range": "",
            "date": "",
            "original_description": general_desc
        })
    
    # Extract event-specific descriptions
    for time_block in data.get("events", []):
        print("*"*80)
        print("Time Block:", time_block)  # Debugging line to check the time block
        time_range = time_block.get("time_range", "")
        for evt in time_block.get("events", []):
            event_name = evt.get("event", "Unnamed Event")
            event_date = evt.get("date", "")
            event_desc = evt.get("description", "")

            # If event description is a string, process it
            if isinstance(event_desc, str):
                event_desc_chunks = chunk_text(event_desc)
                for chunk in event_desc_chunks:
                    descriptions.append({
                        "chunk": chunk,
                        "original_event_name": event_name,
                        "time_range": time_range,
                        "date": event_date,
                        "original_description": event_desc
                    })
            
            # If event description is a dictionary (nested descriptions), process it
            elif isinstance(event_desc, dict):
                for subdesc in event_desc.values():
                    subdesc_chunks = chunk_text(subdesc)
                    for chunk in subdesc_chunks:
                        descriptions.append({
                            "chunk": chunk,
                            "original_event_name": event_name,
                            "time_range": time_range,
                            "date": event_date,
                        })

    return descriptions

# Extract descriptions and print the result
descriptions = extract_descriptions(data)

# Step 4: Store the result in JSON format
output_data = {
    "persona": data["persona"],
    "descriptions": descriptions
}

# Save to a new JSON file
with open("alan_turing_chunks.json", 'w') as outfile:
    json.dump(output_data, outfile, indent=4)

print("Chunks created and saved to 'alan_turing_chunks.json'.")

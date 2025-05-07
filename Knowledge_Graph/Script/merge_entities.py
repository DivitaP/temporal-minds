import json
from collections import defaultdict

def merge_json_to_neo4j(json_data):
    """
    Process JSON data to merge duplicate entities and create Cypher statements for Neo4j
    """
    # Extract or initialize entities and relationships
    if isinstance(json_data, str):
        # If input is a string, parse it
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError:
            print("Invalid JSON format provided")
            return None
    else:
        # Assume it's already a dictionary
        data = json_data
    
    # Extract entities and relationships from various possible keys
    entities = []
    relationships = []
    
    # Look for entities in various possible keys
    for key in data:
        if key.startswith('entities'):
            entities.extend(data[key])
    
    # Look for relationships in various possible keys
    for key in data:
        if key.startswith('relationships'):
            relationships.extend(data[key])
    
    if not entities:
        print("No entities found in the provided JSON")
        return None
    
    # Step 1: Merge duplicate entities
    merged_entities = merge_duplicate_entities(entities)
    
    # Step 2: Generate Cypher statements
    cypher_statements = generate_cypher_statements(merged_entities, relationships)
    
    return {
        "merged_entities": merged_entities,
        "cypher_statements": cypher_statements
    }

def merge_duplicate_entities(entities):
    """
    Merge entities with the same ID and type
    """
    entity_map = {}
    
    for entity in entities:
        entity_id = entity['id']
        entity_type = entity['type']
        key = f"{entity_id}_{entity_type}"
        
        if key not in entity_map:
            entity_map[key] = entity
        else:
            # Merge attributes
            existing_entity = entity_map[key]
            
            # If the label is different, prefer the non-empty one
            if 'label' in entity and (
                'label' not in existing_entity or 
                (not existing_entity['label'] and entity['label'])
            ):
                existing_entity['label'] = entity['label']
            
            # Merge attributes dictionaries
            if 'attributes' in entity:
                if 'attributes' not in existing_entity:
                    existing_entity['attributes'] = {}
                
                for attr_key, attr_value in entity['attributes'].items():
                    # If the attribute doesn't exist or is empty in the existing entity
                    if (attr_key not in existing_entity['attributes'] or 
                        not existing_entity['attributes'][attr_key]):
                        existing_entity['attributes'][attr_key] = attr_value
                    # If both have values, combine them if they're different
                    elif attr_value and attr_value != existing_entity['attributes'][attr_key]:
                        # For description fields, we might want to combine them
                        if attr_key == 'description':
                            existing_entity['attributes'][attr_key] = (
                                f"{existing_entity['attributes'][attr_key]}. {attr_value}"
                            )
                        # For other fields, we could implement different merging strategies
    
    return list(entity_map.values())

def generate_cypher_statements(entities, relationships):
    """
    Generate Cypher statements for Neo4j import
    """
    cypher_statements = []
    
    # Create nodes
    for entity in entities:
        entity_id = entity['id']
        entity_type = entity['type']
        label = entity.get('label', entity_id)
        
        # Build properties string
        properties = {
            'id': entity_id,
            'label': label
        }
        
        # Add attributes if they exist
        if 'attributes' in entity:
            for attr_key, attr_value in entity['attributes'].items():
                properties[attr_key] = attr_value
        
        # Create Cypher statement for node
        properties_str = format_properties(properties)
        cypher_statements.append(
            f"CREATE (:{entity_type} {properties_str});"
        )
    
    # Create unique constraint and index for id property
    cypher_statements.insert(0, "CREATE CONSTRAINT FOR (n) REQUIRE n.id IS UNIQUE;")
    
    # Create relationships
    for rel in relationships:
        source_id = rel['source']
        target_id = rel['target']
        rel_type = rel['type']
        
        # Build properties string for relationship
        properties = {}
        if 'attributes' in rel:
            for attr_key, attr_value in rel['attributes'].items():
                properties[attr_key] = attr_value
        
        properties_str = format_properties(properties) if properties else ""
        
        # Create Cypher statement for relationship
        cypher_statements.append(
            f"MATCH (source {{id: '{source_id}'}}), (target {{id: '{target_id}'}}) "
            f"CREATE (source)-[:{rel_type} {properties_str}]->(target);"
        )
    
    return cypher_statements

def format_properties(props):
    """
    Format a Python dictionary as a Cypher properties string
    """
    property_items = []
    for key, value in props.items():
        if isinstance(value, str):
            escaped_value = value.replace("'", "\\'")
            property_items.append(f"{key}: '{escaped_value}'")
        elif value is None:
            property_items.append(f"{key}: null")
        else:
            property_items.append(f"{key}: {value}")
    
    return "{" + ", ".join(property_items) + "}"


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python script.py input_json_file output_cypher_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Load JSON from file
    try:
        with open(input_file, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)
    
    # Process the data
    result = merge_json_to_neo4j(json_data)
    
    if result:
        # Write Cypher statements to file
        try:
            with open(output_file, 'w') as f:
                for statement in result["cypher_statements"]:
                    f.write(statement + "\n")
            print(f"Cypher statements written to {output_file}")
        except Exception as e:
            print(f"Error writing to output file: {e}")
            sys.exit(1)
import json
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN

# Step 1: Load your chunked JSON
def load_descriptions(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data["descriptions"]

# Step 2: Extract just the text chunks for topic modeling
def extract_chunks(descriptions):
    return [desc["chunk"] for desc in descriptions]

# Step 3: Run BERTopic with a better model and custom clustering
def run_topic_modeling(chunks):
    embedding_model = SentenceTransformer("all-mpnet-base-v2")  # Better embeddings
    hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1, prediction_data=True)  # Less strict clustering
    topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=hdbscan_model,
        language="english",
        verbose=True
    )
    topics, probs = topic_model.fit_transform(chunks)
    return topic_model, topics, probs

# Step 4: Update descriptions with topic numbers and top words
def update_descriptions_with_topics(descriptions, topics, probs, topic_model):
    for i, desc in enumerate(descriptions):
        topic_id = int(topics[i])
        topic_words = topic_model.get_topic(topic_id)
        topic_label = ", ".join([word for word, _ in topic_words[:5]]) if topic_words else "No topic"

        desc["topic_id"] = topic_id
        desc["topic_label"] = topic_label
        desc["probability"] = float(probs[i]) if probs[i] is not None else None
    return descriptions

# Step 5: Save the updated output
def save_topic_results(persona_name, descriptions, output_path):
    output_data = {
        "persona": persona_name,
        "topics": descriptions
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

# === Main pipeline ===
def main():
    persona_name = "Alan Turing"
    input_path = "../JSON/alan_turing_chunks.json"     # Input JSON with "descriptions"
    output_path = "../JSON/alan_turing_topics.json"    # Output with topics added

    descriptions = load_descriptions(input_path)
    chunks = extract_chunks(descriptions)
    topic_model, topics, probs = run_topic_modeling(chunks)
    updated_descriptions = update_descriptions_with_topics(descriptions, topics, probs, topic_model)
    save_topic_results(persona_name, updated_descriptions, output_path)

    print(f"✅ Topic modeling complete. Results saved to: {output_path}")

if __name__ == "__main__":
    main()

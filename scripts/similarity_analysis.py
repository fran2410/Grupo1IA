# scripts/similarity_analysis.py

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from rdflib import Graph, Namespace, RDF, URIRef, Literal
from rdflib.namespace import XSD

# Añadir scripts al path
sys.path.append(os.path.dirname(__file__))

# === Configuración de rutas ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_DIR = os.path.join(BASE_DIR, "..", "papersXML")

# === Modelo de embeddings y KeyBERT ===
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(embed_model)

# === 1. Extraer tópicos con KeyBERT ===
def extract_topics(text, top_n=3):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw[0].replace(" ", "_") for kw in keywords]

# === 2. Similaridad de abstracts ===
def compute_similarity(abstracts_dict):
    abstracts = list(abstracts_dict.values())
    filenames = list(abstracts_dict.keys())
    embeddings = embed_model.encode(abstracts, convert_to_tensor=False)
    sim_matrix = cosine_similarity(embeddings)
    return sim_matrix, filenames

# === 3. Visualizaciones ===
def plot_similarity_matrix(matrix, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap='coolwarm')
    plt.xticks(rotation=90)
    plt.title("Similarity Matrix of Abstracts")
    plt.tight_layout()
    plt.savefig(os.path.join("output_files", "similarity_matrix.png"))
    plt.close()

def plot_dendrogram(matrix, labels):
    distance = 1 - matrix
    linked = linkage(distance, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(linked, labels=labels, leaf_rotation=90)
    plt.title("Dendrogram of Abstract Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join("output_files", "dendrogram.png"))
    plt.close()

# === 4. Crear RDF con tópicos y similitud ===
def generate_rdf_compatible(abstracts_dict, sim_matrix, labels, topics_by_paper):
    g = Graph()
    EX = Namespace("http://example.org/")
    SCHEMA = Namespace("http://schema.org/")
    g.bind("ex", EX)
    g.bind("schema1", SCHEMA)

    paper_uris = {}

    for filename, abstract in abstracts_dict.items():
        paper_id = filename.replace(".xml", "")
        paper_uri = URIRef(f"http://example.org/{paper_id}")
        paper_uris[filename] = paper_uri

        g.add((paper_uri, RDF.type, SCHEMA.ScholarlyArticle))
        g.add((paper_uri, SCHEMA.name, Literal(abstract[:150] + "...")))

        # Keywords como palabras (hasWord)
        for word in abstract.split()[:10]:  # puedes ajustar la lógica
            g.add((paper_uri, EX.hasWord, Literal(word.lower())))

        # Topics
        for topic in topics_by_paper.get(filename, []):
            topic_id = topic.replace(" ", "_")
            topic_uri = URIRef(f"http://example.org/Topic_{topic_id}")
            g.add((topic_uri, RDF.type, SCHEMA.Thing))
            g.add((topic_uri, SCHEMA.name, Literal(topic.replace("_", " "))))
            g.add((paper_uri, EX.belongsToTopic, topic_uri))

    # Similaridades entre papers
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            score = float(sim_matrix[i][j])
            if score > 0.75:
                uri_a = paper_uris[labels[i]]
                uri_b = paper_uris[labels[j]]

                g.add((uri_a, EX.hasSimilarityTo, uri_b))
                g.add((uri_a, EX.similarityScore, Literal(score, datatype=XSD.float)))

    RESULTS_DIR = os.path.join("output_files")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    g.serialize(destination=os.path.join(RESULTS_DIR, "similarity_data.rdf"), format="xml")
    print("✅ RDF compatible con app.py guardado como similarity_data.rdf")



if __name__ == "__main__":
    json_path = os.path.join("output_files", "Text_Extraction_results.json")

    if not os.path.exists(json_path):
        print(f"No se encontró el archivo JSON: {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Construir diccionario de abstracts simulados desde 'words'
    abstracts = {
        filename: ' '.join(entry["words"])
        for filename, entry in data.items()
        if "words" in entry and entry["words"]
    }

    if not abstracts:
        print("No se encontraron abstracts válidos en el JSON.")
        sys.exit(1)

    # Detectar tópicos para cada "abstract"
    topics_by_paper = {filename: extract_topics(text) for filename, text in abstracts.items()}

    sim_matrix, labels = compute_similarity(abstracts)
    plot_similarity_matrix(sim_matrix, labels)
    plot_dendrogram(sim_matrix, labels)
    generate_rdf_compatible(abstracts, sim_matrix, labels, topics_by_paper)

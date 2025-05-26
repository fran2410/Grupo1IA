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
    plt.savefig("similarity_matrix.png")
    plt.close()

def plot_dendrogram(matrix, labels):
    distance = 1 - matrix
    linked = linkage(distance, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(linked, labels=labels, leaf_rotation=90)
    plt.title("Dendrogram of Abstract Similarity")
    plt.tight_layout()
    plt.savefig( "dendrogram.png")
    plt.close()

# === 4. Crear RDF con tópicos y similitud ===
def generate_rdf(abstracts_dict, sim_matrix, labels, topics_by_paper):
    g = Graph()
    EX = Namespace("http://example.org/")
    SCHEMA = Namespace("http://schema.org/")
    g.bind("ex", EX)
    g.bind("schema", SCHEMA)

    paper_uris = {}

    # Añadir papers y tópicos
    for filename, abstract in abstracts_dict.items():
        paper_id = filename.replace(".xml", "")
        paper_uri = URIRef(f"http://example.org/{paper_id}")
        paper_uris[filename] = paper_uri

        g.add((paper_uri, RDF.type, SCHEMA.ScholarlyArticle))
        g.add((paper_uri, SCHEMA.name, Literal(abstract[:100] + "...")))

        # Añadir tópicos extraídos automáticamente con reificación opcional
        topics = topics_by_paper.get(filename, [])
        for topic in topics:
            topic_uri = URIRef(f"http://example.org/Topic_{topic}")
            g.add((topic_uri, RDF.type, SCHEMA.Thing))
            g.add((topic_uri, SCHEMA.name, Literal(topic.replace("_", " "))))
            
            # Enlace reificado para topic (opcional, si luego quieres añadir datos extra)
            topic_relation = URIRef(f"http://example.org/topicRelation_{paper_id}_{topic}")
            g.add((topic_relation, RDF.type, EX.TopicRelation))
            g.add((topic_relation, EX.topicOfPaper, paper_uri))
            g.add((topic_relation, EX.topic, topic_uri))
            
            # Enlace desde paper al nodo de relación para indicar pertenencia (opcional)
            g.add((paper_uri, EX.hasTopicRelation, topic_relation))
            
            # Si no quieres complicar, también puedes dejar sólo:
            # g.add((paper_uri, EX.belongsToTopic, topic_uri))

    # Añadir relaciones de similitud con nodo intermedio para el score
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            score = float(sim_matrix[i][j])
            if score > 0.75:  # Umbral de similitud
                paper_a = paper_uris[labels[i]]
                paper_b = paper_uris[labels[j]]
                
                # Nodo intermediario que representa la similitud
                similarity_node = URIRef(f"http://example.org/similarity_{labels[i]}_{labels[j]}")
                g.add((similarity_node, RDF.type, EX.SimilarityRelation))
                g.add((similarity_node, EX.similarPaper1, paper_a))
                g.add((similarity_node, EX.similarPaper2, paper_b))
                g.add((similarity_node, EX.similarityScore, Literal(score, datatype=XSD.float)))

                # También puedes añadir el enlace directo entre papers si quieres
                g.add((paper_a, EX.hasSimilarityTo, paper_b))
                # Opcionalmente añadir el inverso:
                g.add((paper_b, EX.hasSimilarityTo, paper_a))

    # Guardar RDF
    rdf_path =  "similarity_data.rdf"
    g.serialize(destination=rdf_path, format="xml")
    print(f"RDF generado en: {rdf_path}")


if __name__ == "__main__":
    json_path = os.path.join("Text_Extraction_results.json")

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
    generate_rdf(abstracts, sim_matrix, labels, topics_by_paper)

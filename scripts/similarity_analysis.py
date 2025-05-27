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
from collections import Counter

# === Configuración inicial ===
sys.path.append(os.path.dirname(__file__))  # Añadir scripts al path

# Configuración de rutas de archivos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_DIR = os.path.join(BASE_DIR, "..", "papersXML")

# === Modelos de IA ===
embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo para embeddings de texto
kw_model = KeyBERT(embed_model)  # Modelo para extracción de keywords

# === Mapeo de categorías ===
category_mapping = {
    # ... (mapeo existente de categorías)
}

def classify_paper_category(text, top_keywords=10):
    '''
    Clasifica un paper en una categoría temática usando keywords extraídos.
    Parámetros:
        text (str): Texto completo del paper
        top_keywords (int): Número máximo de keywords a extraer
    Retorna:
        str: Nombre de la categoría asignada
    '''
    # Extracción de keywords con KeyBERT
    keywords = kw_model.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 2), 
        stop_words='english', 
        top_n=top_keywords,
        use_mmr=True,
        diversity=0.5
    )
    
    # Procesamiento de keywords extraídos
    extracted_words = []
    for kw, score in keywords:
        words = kw.lower().replace('-', '_').split()
        extracted_words.extend(words)
    
    # Asignación de categoría por coincidencias
    category_scores = {}
    for category_words, category_name in category_mapping.items():
        matches = sum(1 for word in extracted_words if word in category_words)
        if matches > 0:
            category_scores[category_name] = matches
    
    return max(category_scores.keys(), key=lambda x: category_scores[x], default='General')

def extract_topics_from_clusters(words):
    '''
    Asigna categorías temáticas a cada paper basado en su contenido.
    Parámetros:
        words (dict): Diccionario con textos de los papers
    Retorna:
        dict: Mapeo de papers a categorías asignadas
    '''
    topics_by_paper = {}
    for filename, text in words.items():
        category = classify_paper_category(text)
        topics_by_paper[filename] = category
        print(f"{filename} -> {category}")
    return topics_by_paper

def compute_similarity(words):
    '''
    Calcula matriz de similitud coseno entre abstracts de papers.
    Parámetros:
        words (dict): Diccionario con textos de los papers
    Retorna:
        tuple: Matriz de similitud y lista de identificadores
    '''
    abstracts = list(words.values())
    filenames = list(words.keys())
    embeddings = embed_model.encode(abstracts, convert_to_tensor=False)
    return cosine_similarity(embeddings), filenames

def plot_similarity_matrix(matrix, labels):
    '''Genera y guarda heatmap de matriz de similitud'''
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, 
                cmap='coolwarm', annot=False, fmt='.2f')
    plt.title("Matriz de Similitud entre Abstracts")
    plt.savefig(os.path.join("output_files", "similarity_matrix.png"), dpi=300)
    plt.close()

def plot_dendrogram(matrix, labels):
    '''Genera y guarda dendrograma de agrupamiento jerárquico'''
    distance = 1 - matrix
    linked = linkage(distance, method='ward')
    plt.figure(figsize=(15, 8))
    dendrogram(linked, labels=labels, leaf_rotation=90)
    plt.title("Dendrograma de Similitud entre Papers")
    plt.savefig(os.path.join("output_files", "dendrogram.png"), dpi=300)
    plt.close()

def generate_rdf_compatible(words, sim_matrix, labels, topics_by_paper):
    '''
    Genera grafo RDF con relaciones de similitud y categorías.
    Parámetros:
        words (dict): Textos de los papers
        sim_matrix (np.array): Matriz de similitud
        labels (list): Identificadores de papers
        topics_by_paper (dict): Categorías asignadas
    '''
    g = Graph()
    EX = Namespace("http://example.org/")
    SCHEMA = Namespace("http://schema.org/")
    g.bind("ex", EX)
    g.bind("schema1", SCHEMA)

    # Creación de nodos para papers y temas
    paper_uris = {}
    for filename, abstract in words.items():
        paper_id = filename.replace(".xml", "")
        paper_uri = URIRef(f"http://example.org/{paper_id}")
        paper_uris[filename] = paper_uri
        g.add((paper_uri, RDF.type, SCHEMA.ScholarlyArticle))
        
        # Vinculación con tema principal
        topic = topics_by_paper.get(filename, 'General')
        topic_uri = URIRef(f"http://example.org/Topic_{topic.replace(' ', '_')}")
        g.add((topic_uri, SCHEMA.name, Literal(topic)))
        g.add((paper_uri, EX.belongsToTopic, topic_uri))

    # Creación de relaciones de similitud
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            score = float(sim_matrix[i][j])
            uri_a = paper_uris[labels[i]]
            uri_b = paper_uris[labels[j]]
            g.add((uri_a, EX.hasSimilarityTo, uri_b))
            g.add((uri_a, EX.similarityScore, Literal(score, datatype=XSD.float)))

    # Guardado del grafo RDF
    g.serialize(destination=os.path.join("output_files", "similarity_data.rdf"), format="xml")

if __name__ == "__main__":
    '''Flujo principal de ejecución del análisis'''
    # Carga de datos
    json_path = os.path.join("output_files", "Text_Extraction_results.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Preparación de abstracts
    words = {filename: ' '.join(entry["words"]) for filename, entry in data.items() if "words" in entry}

    # Procesamiento principal
    os.makedirs("output_files", exist_ok=True)
    topics_by_paper = extract_topics_from_clusters(words)
    sim_matrix, labels = compute_similarity(words)
    
    # Generación de resultados
    plot_similarity_matrix(sim_matrix, labels)
    plot_dendrogram(sim_matrix, labels)
    generate_rdf_compatible(words, sim_matrix, labels, topics_by_paper)
    
    print("\nProceso completado. Resultados en 'output_files/'")
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
from collections import Counter

# Añadir scripts al path
sys.path.append(os.path.dirname(__file__))

# === Configuración de rutas ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_DIR = os.path.join(BASE_DIR, "..", "papersXML")

# === Modelo de embeddings y KeyBERT ===
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(embed_model)

# === Mapeo de categorías por clusters de palabras clave ===
category_mapping = {
    # Machine Learning
    ('learning', 'model', 'training', 'algorithm', 'prediction', 'predictive', 'learner', 'parametric', 'nonparametric', 'overfitting', 'generalization', 'optimization', 'hyperparameter'): 'Machine Learning',

    # Deep Learning
    ('neural', 'network', 'deep', 'cnn', 'lstm', 'rnn', 'transformer', 'encoder', 'decoder', 'attention', 'gan', 'autoencoder', 'backpropagation', 'dropout', 'activation', 'convolution'): 'Deep Learning',

    # Supervised Learning
    ('classification', 'classifier', 'supervised', 'regression', 'label', 'decision tree', 'random forest', 'svm', 'support vector', 'linear model', 'logistic', 'ground truth', 'auc', 'precision', 'recall'): 'Supervised Learning',

    # Unsupervised Learning
    ('clustering', 'unsupervised', 'kmeans', 'cluster', 'dbscan', 'hierarchical', 'embedding', 'dimensionality', 'latent', 't-sne', 'manifold', 'density estimation', 'pca'): 'Unsupervised Learning',

    # Reinforcement Learning
    ('reinforcement', 'policy', 'agent', 'q-learning', 'reward', 'environment', 'markov', 'mdp', 'value function', 'exploration', 'exploitation', 'bellman'): 'Reinforcement Learning',

    # Feature Engineering
    ('feature', 'selection', 'extraction', 'reduction', 'engineering', 'dimensionality', 'normalization', 'scaling', 'encoding', 'binarization', 'discretization'): 'Feature Engineering',

    # Bayesian Methods
    ('bayesian', 'probabilistic', 'inference', 'posterior', 'prior', 'likelihood', 'distribution', 'markov chain', 'monte carlo', 'mcmc', 'variational', 'belief', 'graphical model'): 'Bayesian Methods',

    # Artificial Intelligence
    ('artificial', 'intelligence', 'ai', 'intelligent', 'agent', 'autonomy', 'cognitive', 'adaptive', 'symbolic', 'machine reasoning'): 'Artificial Intelligence',

    # Knowledge Systems
    ('reasoning', 'knowledge', 'expert', 'inference', 'ontologies', 'rules', 'facts', 'knowledge base', 'semantic', 'logic'): 'Knowledge Systems',

    # Automated Planning
    ('planning', 'decision', 'heuristic', 'search', 'pathfinding', 'plan', 'goals', 'actions', 'state space'): 'Automated Planning',

    # NLP
    ('text', 'language', 'nlp', 'natural', 'processing', 'semantic', 'syntax', 'sentence', 'linguistic', 'document', 'parsing', 'vocabulary'): 'Natural Language Processing',

    # Text Analysis
    ('sentiment', 'analysis', 'mining', 'extraction', 'opinion', 'topic modeling', 'tf-idf', 'keywords', 'summarization', 'information retrieval'): 'Text Analysis',

    # Machine Translation
    ('translation', 'machine', 'linguistic', 'syntax', 'semantic', 'translating', 'bilingual', 'alignment', 'dictionary'): 'Machine Translation',

    # Text Representation
    ('tokenization', 'embedding', 'word2vec', 'bert', 'vectorization', 'contextual', 'glove', 'elmo', 'transformers'): 'Text Representation',

    # Computer Vision
    ('image', 'vision', 'visual', 'recognition', 'detection', 'perception', 'scene', 'photograph', 'multiview', 'augmentation'): 'Computer Vision',

    # Image Analysis
    ('object', 'face', 'pattern', 'segmentation', 'edge', 'contour', 'feature map', 'bounding box', 'masking', 'histogram'): 'Image Analysis',

    # Video Analysis
    ('video', 'motion', 'tracking', 'frame', 'temporal', 'optical flow', 'keyframe', 'stream'): 'Video Analysis',

    # 3D Vision
    ('camera', 'stereo', 'depth', 'calibration', 'point cloud', '3d reconstruction', 'lidar', 'mesh'): '3D Vision',

    # Speech Processing
    ('speech', 'audio', 'sound', 'recognition', 'phoneme', 'asr', 'speaker', 'transcription', 'prosody'): 'Speech Processing',

    # Audio Signal Processing
    ('signal', 'frequency', 'waveform', 'acoustic', 'fft', 'spectrogram', 'noise reduction', 'pitch', 'echo'): 'Audio Signal Processing',

    # Healthcare
    ('medical', 'health', 'clinical', 'patient', 'disease', 'epidemiology', 'healthcare', 'symptom', 'screening'): 'Healthcare',

    # Medical Systems
    ('diagnosis', 'treatment', 'therapeutic', 'medicine', 'radiology', 'monitoring', 'biomedical', 'prognosis', 'intervention'): 'Medical Systems',

    # Bioinformatics
    ('genome', 'dna', 'bioinformatics', 'protein', 'rna', 'sequencing', 'genomic', 'mutation', 'alignment', 'biological'): 'Bioinformatics',

    # Neuroscience
    ('brain', 'neuro', 'eeg', 'cognitive', 'fmri', 'neuron', 'psychology', 'neuroscience', 'mental'): 'Neuroscience & Cognitive Science',

    # Data Science
    ('data', 'analytics', 'statistics', 'mining', 'big', 'science', 'pattern', 'processing', 'insight'): 'Data Science',

    # Data Analytics
    ('visualization', 'exploratory', 'statistical', 'dashboard', 'plot', 'chart', 'analytics', 'bar graph', 'histogram'): 'Data Analytics',

    # Time Series
    ('time', 'series', 'temporal', 'forecast', 'trend', 'seasonality', 'lag', 'signal', 'autocorrelation'): 'Time Series Analysis',

    # Software Engineering
    ('architecture', 'design', 'implementation', 'performance', 'maintainability', 'refactoring', 'testing', 'debugging'): 'Software Engineering',

    # Systems & Tech
    ('system', 'software', 'technology', 'application', 'deployment', 'platform', 'infrastructure', 'engineering'): 'Systems & Technology',

    # Network Systems
    ('network', 'distributed', 'cloud', 'computing', 'scalability', 'latency', 'bandwidth', 'throughput'): 'Network Systems',

    # Embedded Systems
    ('real-time', 'embedded', 'firmware', 'runtime', 'microcontroller', 'low-power', 'scheduling'): 'Embedded Systems',

    # Parallel Computing
    ('parallel', 'concurrent', 'thread', 'multicore', 'synchronization', 'gpu', 'mpi', 'openmp'): 'Parallel Computing',

    # Cybersecurity
    ('security', 'privacy', 'encryption', 'cyber', 'firewall', 'cryptography', 'confidentiality', 'integrity', 'authentication'): 'Cybersecurity',

    # IoT
    ('iot', 'sensor', 'wireless', 'mobile', 'networked', 'embedded', 'connectivity', 'smart', 'device', 'edge'): 'IoT & Mobile Systems',

    # Robotics
    ('robot', 'robotics', 'autonomous', 'control', 'servo', 'motor', 'feedback', 'controller', 'planner', 'localization'): 'Robotics',

    # HCI
    ('interface', 'user', 'interaction', 'usability', 'ui', 'ux', 'design', 'human-centered', 'input', 'accessibility'): 'Human-Computer Interaction',

    # VR/AR
    ('vr', 'ar', 'virtual', 'augmented', 'immersive', 'headset', 'hologram', 'reality'): 'Virtual & Augmented Reality',

    # Ethics
    ('ethics', 'bias', 'fairness', 'transparency', 'accountability', 'discrimination', 'explainability'): 'AI Ethics',

    # Education
    ('education', 'learning', 'teaching', 'student', 'e-learning', 'mooc', 'edtech', 'tutor', 'curriculum'): 'Educational Technology',

    # Optimization
    ('optimization', 'constraint', 'linear', 'solver', 'objective', 'gradient', 'minimization', 'convergence', 'lp', 'qp'): 'Optimization',

    # Operations Research
    ('scheduling', 'routing', 'resource', 'allocation', 'queue', 'inventory', 'logistics'): 'Operations Research',

    # Theoretical CS
    ('complexity', 'algorithmic', 'proof', 'theorem', 'hardness', 'automata', 'turing', 'np-complete'): 'Theoretical Computer Science',

    # Graph Theory
    ('graph', 'tree', 'node', 'edge', 'path', 'cycle', 'bipartite', 'clique', 'adjacency'): 'Graph Theory',

    # Blockchain
    ('blockchain', 'ledger', 'crypto', 'decentralized', 'consensus', 'smart contract', 'mining', 'ethereum'): 'Blockchain Technology',

    # Quantum
    ('quantum', 'qubit', 'entanglement', 'superposition', 'quantization', 'qiskit', 'quantum gate', 'measurement'): 'Quantum Computing',
}

# === 1. Clasificar paper en categoría usando KeyBERT ===
def classify_paper_category(text, top_keywords=10):
    """
    Extrae keywords con KeyBERT y las mapea a las categorías definidas
    """
    # Extraer keywords con KeyBERT
    keywords = kw_model.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 2), 
        stop_words='english', 
        top_n=top_keywords,
        use_mmr=True,
        diversity=0.5
    )
    
    # Convertir a lista de palabras simples
    extracted_words = []
    for kw, score in keywords:
        # Dividir frases en palabras individuales y limpiar
        words = kw.lower().replace('-', '_').split()
        extracted_words.extend(words)
    
    # Contar coincidencias con cada categoría
    category_scores = {}
    for category_words, category_name in category_mapping.items():
        matches = sum(1 for word in extracted_words if word in category_words)
        if matches > 0:
            category_scores[category_name] = matches
    
    # Retornar categoría con más coincidencias
    if category_scores:
        best_category = max(category_scores.keys(), key=lambda x: category_scores[x])
        return best_category
    else:
        return 'General'  # Categoría por defecto


# === 3. Procesamiento principal de tópicos ===
def extract_topics_from_clusters(words):
    """
    Clasifica cada paper y asigna los 3 tópicos más generales de su categoría
    """
    topics_by_paper = {}
    
    for filename, text in words.items():
        # Clasificar paper en categoría
        category = classify_paper_category(text)
        
        # Obtener tópicos generales de la categoría
        topics_by_paper[filename] = category
        
        print(f"{filename} -> {category}")
    
    return topics_by_paper

# === 4. Similaridad de abstracts ===
def compute_similarity(words):
    abstracts = list(words.values())
    filenames = list(words.keys())
    embeddings = embed_model.encode(abstracts, convert_to_tensor=False)
    sim_matrix = cosine_similarity(embeddings)
    
    return sim_matrix, filenames

# === 5. Visualizaciones ===
def plot_similarity_matrix(matrix, labels):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap='coolwarm', 
                annot=False, fmt='.2f', cbar_kws={'label': 'Cosine Similarity'})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Similarity Matrix of Abstracts", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join("output_files", "similarity_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_dendrogram(matrix, labels):
    distance = 1 - matrix
    linked = linkage(distance, method='ward')
    plt.figure(figsize=(15, 8))
    dendrogram(linked, labels=labels, leaf_rotation=90, leaf_font_size=10)
    plt.title("Dendrogram of Abstract Similarity", fontsize=16)
    plt.xlabel("Papers", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join("output_files", "dendrogram.png"), dpi=300, bbox_inches='tight')
    plt.close()

# === 6. Crear RDF con tópicos y similitud ===
def generate_rdf_compatible(words, sim_matrix, labels, topics_by_paper):
    g = Graph()
    EX = Namespace("http://example.org/")
    SCHEMA = Namespace("http://schema.org/")
    g.bind("ex", EX)
    g.bind("schema1", SCHEMA)

    paper_uris = {}

    for filename, abstract in words.items():
        paper_id = filename.replace(".xml", "")
        paper_uri = URIRef(f"http://example.org/{paper_id}")
        paper_uris[filename] = paper_uri

        g.add((paper_uri, RDF.type, SCHEMA.ScholarlyArticle))
        g.add((paper_uri, SCHEMA.name, Literal(abstract[:150] + "...")))

        topic = topics_by_paper.get(filename, [])
        topic_id = topic.replace(" ", "_")
        topic_uri = URIRef(f"http://example.org/Topic_{topic_id}")
        g.add((topic_uri, RDF.type, SCHEMA.Thing))
        g.add((topic_uri, SCHEMA.name, Literal(topic.replace("_", " "))))
        g.add((paper_uri, EX.belongsToTopic, topic_uri))

    # Similaridades entre papers (solo las más altas)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            score = float(sim_matrix[i][j])
            uri_a = paper_uris[labels[i]]
            uri_b = paper_uris[labels[j]]

            g.add((uri_a, EX.hasSimilarityTo, uri_b))
            g.add((uri_a, EX.similarityScore, Literal(score, datatype=XSD.float)))

    RESULTS_DIR = os.path.join("output_files")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    g.serialize(destination=os.path.join(RESULTS_DIR, "similarity_data.rdf"), format="xml")
    print("RDF compatible con app.py guardado como similarity_data.rdf")


if __name__ == "__main__":
    json_path = os.path.join("output_files", "Text_Extraction_results.json")

    if not os.path.exists(json_path):
        print(f"No se encontró el archivo JSON: {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Construir diccionario de abstracts simulados desde 'words'
    words = {
        filename: ' '.join(entry["words"])
        for filename, entry in data.items()
        if "words" in entry and entry["words"]
    }

    if not words:
        print("No se encontraron words válidos en el JSON.")
        sys.exit(1)

    print(f"Procesando {len(words)} papers...")

    # Crear directorio de salida
    os.makedirs("output_files", exist_ok=True)

    # Detectar tópicos usando clustering de categorías
    print("\nClasificando papers por categorías...")
    topics_by_paper = extract_topics_from_clusters(words)

    # Computar similaridad
    print("\nCalculando matriz de similaridad...")
    sim_matrix, labels = compute_similarity(words)

    # Generar visualizaciones
    print("\nGenerando visualizaciones...")
    plot_similarity_matrix(sim_matrix, labels)
    plot_dendrogram(sim_matrix, labels)

    # Generar RDF
    print("\n🔧 Generando archivo RDF...")
    generate_rdf_compatible(words, sim_matrix, labels, topics_by_paper)

    print("\nProceso completado. Archivos generados en 'output_files/':")
    print("  - similarity_matrix.png")
    print("  - dendrogram.png")
    print("  - category_distribution.png")
    print("  - similarity_data.rdf")
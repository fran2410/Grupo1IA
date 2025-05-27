import os
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.plugins.sparql import prepareQuery
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(
    page_title="Explorador RDF Académico",
    page_icon="📚",
    layout="wide"
)

SCHEMA = Namespace("http://schema.org/")
OWL = Namespace("http://www.w3.org/2002/07/owl#")
EX = Namespace("http://example.org/")

@st.cache_data
def load_custom_rdf():
    '''Carga el grafo RDF principal desde el archivo XML especificado'''
    g = Graph()
    try:
        g.parse("output_files/knowledge_graph_linked.rdf", format="xml")
        g.bind("schema", SCHEMA)
        g.bind("owl", OWL)
        g.bind("ex", EX)
        return g
    except Exception as e:
        st.error(f"Error cargando el RDF: {e}")
        return Graph()

@st.cache_data
def load_similarity_rdf():
    '''Carga el grafo RDF de similitudes y topics desde archivo XML'''
    g = Graph()
    try:
        g.parse("output_files/similarity_data.rdf", format="xml")
        g.bind("schema", SCHEMA)
        g.bind("ex", EX)
        return g
    except Exception as e:
        st.warning(f"Error cargando el RDF de similarity: {e}")
        return Graph()

def execute_sparql_query(graph, query_string):
    '''Ejecuta una consulta SPARQL y devuelve los resultados como lista'''
    try:
        results = graph.query(query_string)
        return list(results)
    except Exception as e:
        st.error(f"Error en la consulta SPARQL: {e}")
        return []

@st.cache_data
def create_networkx_graph(_rdf_graph):
    '''Convierte el grafo RDF principal en un grafo NetworkX con nodos y relaciones'''
    G = nx.Graph()
    query = """
    PREFIX schema: <http://schema.org/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?subject ?type ?label WHERE {
        ?subject a ?type .
        OPTIONAL { ?subject schema:name ?label }
    }
    """
    results = execute_sparql_query(_rdf_graph, query)
    for row in results:
        node_id = str(row.subject).split('/')[-1]
        node_type = str(row.type).split('/')[-1]
        label = str(row.label) if row.label else node_id
        G.add_node(node_id, type=node_type, label=label)

    relations_queries = [
        ("autor-paper", "schema:author", "author"),
        ("acknowledges", "schema:acknowledges", "acknowledges"),
        ("afiliación", "schema:affiliation", "affiliation"),
        ("funder", "schema:funder", "funder")
    ]
    
    for name, predicate, relation in relations_queries:
        query = f"PREFIX schema: <http://schema.org/> SELECT ?s ?o WHERE {{ ?s {predicate} ?o . }}"
        relations = execute_sparql_query(_rdf_graph, query)
        for s, o in relations:
            source = str(s).split('/')[-1]
            target = str(o).split('/')[-1]
            G.add_edge(source, target, relation=relation)
    
    return G

@st.cache_data
def create_similarity_networkx_graph(_rdf_graph, _similarity_graph):
    '''Crea un grafo NetworkX para relaciones de similitud entre papers'''
    G = nx.Graph()
    if not _similarity_graph:
        return G

    query_papers = """
    PREFIX schema: <http://schema.org/>
    SELECT ?paper ?title WHERE {
        ?paper a schema:ScholarlyArticle ;
               schema:name ?title .
    }
    """
    papers = execute_sparql_query(_rdf_graph, query_papers)
    for paper, title in papers:
        paper_id = str(paper).split('/')[-1]
        G.add_node(paper_id, type="ScholarlyArticle", label=str(title))

    query_topics = """
    PREFIX ex: <http://example.org/>
    SELECT DISTINCT ?topic ?name WHERE {
        ?paper ex:belongsToTopic ?topic .
        ?topic schema:name ?name .
    }
    """
    topics = execute_sparql_query(_similarity_graph, query_topics)
    for topic, name in topics:
        topic_id = str(topic).split('/')[-1]
        G.add_node(topic_id, type="Topic", label=str(name))

    query_relations = """
    PREFIX ex: <http://example.org/>
    SELECT ?paper ?topic WHERE {
        ?paper ex:belongsToTopic ?topic .
    }
    """
    relations = execute_sparql_query(_similarity_graph, query_relations)
    for paper, topic in relations:
        paper_id = str(paper).split('/')[-1]
        topic_id = str(topic).split('/')[-1]
        G.add_edge(paper_id, topic_id, relation="belongsToTopic")
    
    return G

def create_agraph_visualization(G):
    '''Genera nodos y edges para visualización interactiva del grafo principal'''
    node_config = {
        'ScholarlyArticle': {'color': "#DB1C1C", 'size': 25, 'shape': 'square'},
        'Person': {'color': "#10AA0B", 'size': 20, 'shape': 'dot'},
        'Organization': {'color': "#7B00E0", 'size': 30, 'shape': 'triangle'},
        'Acknowledgments': {'color': "#EEFF00", 'size': 20, 'shape': 'dot'},
        'unknown': {'color': '#999999', 'size': 15, 'shape': 'dot'}
    }
    nodes = []
    for node_id, data in G.nodes(data=True):
        node_type = data.get('type', 'unknown')
        label = data.get('label', node_id)
        config = node_config.get(node_type, node_config['unknown'])
        nodes.append(Node(
            id=node_id,
            label=label[:20] + "..." if len(label) > 20 else label,
            size=config['size'],
            color=config['color'],
            shape=config['shape'],
            title=f"{label} ({node_type})"
        ))
    
    edges = []
    for src, dst, data in G.edges(data=True):
        relation = data.get('relation', 'related')
        edge_color = {
            'author': '#DB1C1C', 
            'acknowledges': '#EEFF00',
            'affiliation': '#7B00E0',
            'funder': '#7B00E0'
        }.get(relation, '#888888')
        edges.append(Edge(
            source=src,
            target=dst,
            label=relation,
            color=edge_color,
            title=f"Relación: {relation}"
        ))
    
    return nodes, edges

def create_similarity_agraph_visualization(G):
    '''Genera nodos y edges para visualización de relaciones de similitud'''
    node_config = {
        'ScholarlyArticle': {'color': "#DB1C1C", 'size': 35, 'shape': 'dot'},
        'Topic': {'color': "#0713B6", 'size': 25, 'shape': 'diamond'}
    }
    nodes = []
    for node_id, data in G.nodes(data=True):
        node_type = data.get('type', 'unknown')
        label = data.get('label', node_id)
        config = node_config.get(node_type, {'color': '#999', 'size': 15})
        nodes.append(Node(
            id=node_id,
            label=label[:20] + "..." if len(label) > 20 else label,
            size=config['size'],
            color=config['color'],
            shape=config.get('shape', 'dot'),
            title=f"{label} ({node_type})"
        ))
    
    edges = []
    for src, dst, data in G.edges(data=True):
        edges.append(Edge(
            source=src,
            target=dst,
            label="belongsToTopic",
            color="#0713B6",
            title="Relación: belongsToTopic"
        ))
    
    return nodes, edges

def get_agraph_config():
    '''Configuración visual para el componente agraph'''
    return Config(
        width=800,
        height=600,
        directed=False,
        physics=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        maxZoom=2,
        minZoom=0.1
    )

def get_paper_details(rdf_graph, similarity_graph, paper_id):
    '''Obtiene metadatos detallados de un paper específico'''
    query = f"""
    PREFIX schema: <http://schema.org/>
    SELECT ?title ?abstract ?date ?authorName WHERE {{
        ex:{paper_id} schema:name ?title ;
                     schema:abstract ?abstract ;
                     schema:datePublished ?date ;
                     schema:author ?author .
        ?author schema:name ?authorName .
    }}
    """
    results = execute_sparql_query(rdf_graph, query)
    if not results:
        return None
    
    details = {
        'title': str(results[0].title),
        'abstract': str(results[0].abstract),
        'date': str(results[0].date),
        'authors': list(set([str(r.authorName) for r in results]))
    }
    
    topic_query = f"""
    PREFIX ex: <http://example.org/>
    SELECT ?topic WHERE {{ ex:{paper_id} ex:belongsToTopic ?topic . }}
    """
    topic = execute_sparql_query(similarity_graph, topic_query)
    details['topic'] = str(topic[0].topic).split('/')[-1] if topic else None
    
    return details

def get_similarity_score(similarity_graph, paper1, paper2):
    '''Obtiene el puntaje de similitud entre dos papers'''
    query = f"""
    PREFIX ex: <http://example.org/>
    SELECT ?score WHERE {{
        {{ ex:{paper1} ex:hasSimilarityTo ex:{paper2} ; ex:similarityScore ?score . }}
        UNION
        {{ ex:{paper2} ex:hasSimilarityTo ex:{paper1} ; ex:similarityScore ?score . }}
    }}
    """
    results = execute_sparql_query(similarity_graph, query)
    return float(results[0].score) if results else None

def main():
    '''Función principal que configura la interfaz de Streamlit'''
    st.title("Explorador de Publicaciones Académicas")
    
    if 'rdf_graph' not in st.session_state:
        st.session_state.rdf_graph = load_custom_rdf()
        st.session_state.similarity_graph = load_similarity_rdf()
        st.session_state.nx_graph = create_networkx_graph(st.session_state.rdf_graph)
        st.session_state.sim_graph = create_similarity_networkx_graph(
            st.session_state.rdf_graph, st.session_state.similarity_graph)
    
    section = st.sidebar.selectbox(
        "Secciones:",
        ["Grafo Interactivo", "Análisis de Papers", "Comparación", "Consultas SPARQL"]
    )
    
    if section == "Grafo Interactivo":
        st.header("Visualización de Grafos")
        graph_type = st.radio("Tipo de Grafo", ["Principal", "Similitud"])
        
        if graph_type == "Principal":
            nodes, edges = create_agraph_visualization(st.session_state.nx_graph)
        else:
            nodes, edges = create_similarity_agraph_visualization(st.session_state.sim_graph)
        
        agraph(nodes, edges, get_agraph_config())
    
    elif section == "Análisis de Papers":
        st.header("Análisis Detallado")
        papers = execute_sparql_query(st.session_state.rdf_graph, 
            "PREFIX schema: <http://schema.org/> SELECT ?paper ?title WHERE { ?paper a schema:ScholarlyArticle ; schema:name ?title . }")
        selected = st.selectbox("Seleccionar Paper", [f"{str(t)} ({str(p).split('/')[-1]})" for p, t in papers])
        paper_id = selected.split('(')[-1].strip(')')
        details = get_paper_details(st.session_state.rdf_graph, st.session_state.similarity_graph, paper_id)
        if details:
            st.subheader(details['title'])
            st.write(f"**Fecha:** {details['date']}")
            st.write(f"**Topic:** {details['topic']}")
            st.write("**Autores:** " + ", ".join(details['authors']))
            st.expander("Abstract").write(details['abstract'])
    
    elif section == "Comparación":
        st.header("Comparación de Papers")
        papers = execute_sparql_query(st.session_state.rdf_graph, 
            "PREFIX schema: <http://schema.org/> SELECT ?paper ?title WHERE { ?paper a schema:ScholarlyArticle ; schema:name ?title . }")
        paper_list = [f"{str(t)} ({str(p).split('/')[-1]})" for p, t in papers]
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.selectbox("Paper 1", paper_list)
        with col2:
            p2 = st.selectbox("Paper 2", [p for p in paper_list if p != p1])
        p1_id = p1.split('(')[-1].strip(')')
        p2_id = p2.split('(')[-1].strip(')')
        score = get_similarity_score(st.session_state.similarity_graph, p1_id, p2_id)
        if score:
            st.metric("Similitud", f"{score*100:.2f}%")
    
    elif section == "Consultas SPARQL":
        st.header("Consulta Personalizada")
        query = st.text_area("Ingrese su consulta SPARQL")
        if st.button("Ejecutar"):
            results = execute_sparql_query(st.session_state.rdf_graph, query)
            if results:
                st.dataframe(pd.DataFrame([{k: str(v) for k, v in r.asdict().items()} for r in results]))

if __name__ == "__main__":
    main()
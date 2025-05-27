import os
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.plugins.sparql import prepareQuery
from streamlit_agraph import agraph, Node, Edge, Config

# Configuración de la página
st.set_page_config(
    page_title="Explorador RDF Académico",
    page_icon="📚",
    layout="wide"
)

# Definir namespaces globales
SCHEMA = Namespace("http://schema.org/")
OWL = Namespace("http://www.w3.org/2002/07/owl#")
EX = Namespace("http://example.org/")

# Función para cargar el RDF personalizado
@st.cache_data
def load_custom_rdf():
    """Carga el grafo RDF personalizado"""
    g = Graph()
    try:
        # Asegúrate de colocar tu archivo RDF en el mismo directorio o especificar la ruta correcta
        g.parse("output_files/knowledge_graph_linked.rdf", format="xml")
        
        # Bind namespaces para mejor visualización
        g.bind("schema", SCHEMA)
        g.bind("owl", OWL)
        g.bind("ex", EX)
        
        return g
    except Exception as e:
        st.error(f"Error cargando el RDF: {e}")
        return Graph()

# Función para cargar el RDF de similarity
@st.cache_data
def load_similarity_rdf():
    """Carga el grafo RDF de similarity y topics"""
    g = Graph()
    try:
        # Cargar el archivo RDF de similarity
        
        g.parse("output_files/similarity_data.rdf", format="xml")
        
        # Bind namespaces
        g.bind("schema", SCHEMA)
        g.bind("ex", EX)
        
        return g
    except Exception as e:
        st.warning(f"Error cargando el RDF de similarity: {e}. Funcionalidad de similarity no disponible.")
        return Graph()

# Función para ejecutar consultas SPARQL
def execute_sparql_query(graph, query_string):
    """Ejecuta una consulta SPARQL y retorna los resultados"""
    try:
        print(f"Ejecutando consulta SPARQL: {query_string}")
        results = graph.query(query_string)
        return list(results)
    except Exception as e:
        st.error(f"Error en la consulta SPARQL: {e}")
        return []

# Función para crear el grafo NetworkX desde RDF (grafo completo)
@st.cache_data
def create_networkx_graph(_rdf_graph):
    """Convierte el grafo RDF a NetworkX para visualización"""
    G = nx.Graph()
    
    # Consulta para obtener todos los nodos y sus tipos
    query = """
    PREFIX schema: <http://schema.org/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?subject ?type ?label WHERE {
        ?subject a ?type .
        OPTIONAL { 
            ?subject schema:name ?label 
        }
    }
    """
    
    results = execute_sparql_query(_rdf_graph, query)
    
    # Agregar nodos del grafo principal
    for row in results:
        node_id = str(row.subject).split('/')[-1]
        node_type = str(row.type).split('/')[-1]
        label = str(row.label) if row.label else node_id
        
        G.add_node(node_id, type=node_type, label=label)
    
   
    # Consulta para obtener relaciones autor-paper
    relations_query = """
    PREFIX schema: <http://schema.org/>
    SELECT ?paper ?author WHERE {
        ?paper schema:author ?author .
    }
    """
    
    relations = execute_sparql_query(_rdf_graph, relations_query)
    
    # Agregar edges para relaciones paper-author
    for row in relations:
        paper_id = str(row.paper).split('/')[-1]
        author_id = str(row.author).split('/')[-1]
        G.add_edge(paper_id, author_id, relation="author")
    
    # Consulta para obtener relaciones acknowledges
    acknowledge_query = """
    PREFIX schema: <http://schema.org/>
    SELECT ?paper ?acknowledges WHERE {
        ?paper schema:acknowledges ?acknowledges .
    }
    """
    
    akn = execute_sparql_query(_rdf_graph, acknowledge_query)
    
    # Agregar edges para relaciones paper-acknowledges
    for row in akn:
        if "ack_person" in str(row.acknowledges):
            paper_id = str(row.paper).split('/')[-1]
            acknowledges_id = str(row.acknowledges).split('/')[-1]
            G.add_edge(paper_id, acknowledges_id, relation="acknowledges")
    
    # Consulta para relaciones autor-organización
    org_query = """
    PREFIX schema: <http://schema.org/>
    SELECT ?author ?org WHERE {
        ?author schema:affiliation ?org .
    }
    """
    
    org_relations = execute_sparql_query(_rdf_graph, org_query)
    
    for row in org_relations:
        author_id = str(row.author).split('/')[-1]
        org_id = str(row.org).split('/')[-1]
        G.add_edge(author_id, org_id, relation="affiliation")
    
    # Consulta para relaciones paper-funder
    funder_query = """
    PREFIX schema: <http://schema.org/>
    SELECT ?paper ?funder WHERE {
        ?paper schema:funder ?funder .
    }
    """
    
    funder_relations = execute_sparql_query(_rdf_graph, funder_query)
    
    for row in funder_relations:
        paper_id = str(row.paper).split('/')[-1]
        funder_id = str(row.funder).split('/')[-1]
        G.add_edge(paper_id, funder_id, relation="funder")
    
    return G

# Nueva función para crear el grafo de similarity
@st.cache_data
def create_similarity_networkx_graph(_rdf_graph, _similarity_graph):
    """Crea un grafo NetworkX que solo muestra papers y sus similarity scores"""
    G = nx.Graph()
    
    if not _similarity_graph:
        return G
    
    # Primero obtener todos los papers del grafo principal
    papers_query = """
    PREFIX schema: <http://schema.org/>
    SELECT ?paper ?title WHERE {
        ?paper a schema:ScholarlyArticle ;
               schema:name ?title .
    }
    """
    
    papers_results = execute_sparql_query(_rdf_graph, papers_query)
    
    # Agregar nodos de papers
    for row in papers_results:
        paper_id = str(row.paper).split('/')[-1]
        paper_title = str(row.title)
        G.add_node(paper_id, type="ScholarlyArticle", label=paper_title)
    
     # Agregar topics desde el grafo de similarity
    topic_query = """
    PREFIX schema: <http://schema.org/>
    PREFIX ex: <http://example.org/>
    SELECT DISTINCT ?topic ?topicName WHERE {
        ?paper ex:belongsToTopic ?topic .
        ?topic schema:name ?topicName .
    }
    """
    
    topic_results = execute_sparql_query(_similarity_graph, topic_query)
    
    for row in topic_results:
        topic_id = str(row.topic).split('/')[-1]
        topic_name = str(row.topicName)
        G.add_node(topic_id, type="Topic", label=topic_name)
     
    # Agregar relaciones paper-topic desde el grafo de similarity
    paper_topic_query = """
    PREFIX schema: <http://schema.org/>
    PREFIX ex: <http://example.org/>
    SELECT ?paper ?topic WHERE {
        ?paper ex:belongsToTopic ?topic .
    }
    """
    
    paper_topic_relations = execute_sparql_query(_similarity_graph, paper_topic_query)
    for row in paper_topic_relations:
        paper_id = str(row.paper).split('/')[-1]
        topic_id = str(row.topic).split('/')[-1]
        if paper_id in G.nodes() and topic_id in G.nodes():
            G.add_edge(paper_id, topic_id, relation="belongsToTopic")
    
    return G

# Función para crear visualización interactiva con streamlit-agraph (grafo completo)
def create_agraph_visualization(G):
    """Crea un grafo interactivo usando streamlit-agraph"""
    
    # Configurar colores y tamaños por tipo
    node_config = {
        'ScholarlyArticle': {'color': "#DB1C1C", 'size': 25, 'shape': 'square'},
        'Person': {'color': "#10AA0B", 'size': 20, 'shape': 'dot'}, 
        'Organization': {'color': "#7B00E0", 'size': 30, 'shape': 'triangle'},
        'Acknowledgments': {'color': "#EEFF00", 'size': 20, 'shape': 'dot'},
        'Topic': {'color': "#FF8C00", 'size': 35, 'shape': 'diamond'},
        'unknown': {'color': '#999999', 'size': 15, 'shape': 'dot'}
    }
    
    # Crear nodos para agraph
    nodes = []
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        node_type = node_data.get('type', 'unknown')
        label = node_data.get('label', node_id)
        
        config = node_config.get(node_type, node_config['unknown'])
        
        # Crear tooltip con información adicional
        title = f"{label} ({node_type})"
        
        nodes.append(Node(
            id=node_id,
            label=label[:20] + "..." if len(label) > 20 else label,
            size=config['size'],
            color=config['color'],
            shape=config['shape'],
            title=title,
            font={'size': 14, 'color': "#C0C0C0", 'face': "Arial", 'align': "center"},
        ))
    
    # Crear edges para agraph
    edges = []
    for edge in G.edges(data=True):
        source, target, edge_data = edge
        relation = edge_data.get('relation', 'related')
        
        # Configurar color del edge según el tipo de relación
        edge_color = {
            'author': '#DB1C1C',
            'acknowledges': '#EEFF00',
            'funder': '#7B00E0',
            'belongsToTopic': '#FF8C00'
        }.get(relation, '#888888')
        
        edges.append(Edge(
            source=source,
            target=target,
            label=relation,
            color=edge_color,
            width=2,
            font={'size': 9, 'color': "#C0C0C0",'face': "Arial", 'align': "center",  'strokeWidth': 0},
            title=f"Relación: {relation}"
        ))
    
    return nodes, edges

# Nueva función para crear visualización del grafo de similarity
def create_similarity_agraph_visualization(G):
    """Crea un grafo interactivo para similarity scores"""
    node_config = {
        'ScholarlyArticle': {'color': "#DB1C1C", 'size': 35, 'shape': 'dot'},
        'Topic': {'color': "#0713B6", 'size': 25, 'shape': 'diamond'},
        'unknown': {'color': '#999999', 'size': 15, 'shape': 'dot'}
    }
    # Crear nodos para agraph (solo papers)
    nodes = []
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        node_type = node_data.get('type', 'unknown')
        label = node_data.get('label', node_id)
        
        config = node_config.get(node_type, node_config['unknown'])
        
        # Crear tooltip con información adicional
        title = f"{label} ({node_type})"
        
        nodes.append(Node(
            id=node_id,
            label=label[:20] + "..." if len(label) > 20 else label,
            size=config['size'],
            color=config['color'],
            shape=config['shape'],
            title=title,
            font={'size': 12, 'color': "#FFFFFF", 'face': "Arial", 'align': "center"},
        ))
    
    # Crear edges para agraph con similarity scores
    edges = []
    for edge in G.edges(data=True):
        source, target, edge_data = edge
        edges.append(Edge(
        source=source,
        target=target,
        label="belongsToTopic",
        color="#0713B6",
        width=2,
        font={'size': 9, 'color': "#C0C0C0",'face': "Arial", 'align': "center",  'strokeWidth': 0},
        title=f"Relación: belongsToTopic"
        ))
    return nodes, edges

# Función para configurar agraph
def get_agraph_config():
    """Configuración para el grafo agraph"""
    return Config(
        width=800,
        height=600,
        directed=False,
        physics=True,
        stabilize=True,
        fit =True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={'labelProperty': 'label'},
        link={'labelProperty': 'label', 'renderLabel': True},
        maxZoom=2,
        minZoom=0.1,
        staticGraphWithDragAndDrop=False,
        staticGraph=False,
        stabilization={'enabled': True, 'iterations': 100},
        interaction={'dragNodes': True, 'dragView': True, 'zoomView': True},
        layout={'randomSeed': 42}
    )

# Función para obtener detalles de un paper
def get_paper_details(rdf_graph, similarity_graph, paper_id):
    """Obtiene los detalles de un paper específico usando SPARQL"""
    query = f"""
    PREFIX schema: <http://schema.org/>
    PREFIX ex: <http://example.org/>
    
    SELECT ?title ?abstract ?date ?authorName ?orgName ?funderName WHERE {{
        ex:{paper_id} schema:name ?title ;
                     schema:abstract ?abstract ;
                     schema:datePublished ?date ;
                     schema:author ?author .
        ?author schema:name ?authorName.
        
        OPTIONAL {{
            ex:{paper_id} schema:funder ?funder .
            ?funder schema:name ?funderName .
        }}
    }}
    """
    
    results = execute_sparql_query(rdf_graph, query)
    if not results:
        return None
    
    # Procesar resultados
    paper_info = {
        'title': str(results[0].title),
        'abstract': str(results[0].abstract),
        'date': str(results[0].date),
        'authors': [],
        'organizations': [],
        'funders': [],
        'topic': None
    }
    
    for row in results:
        author_info = {
            'name': str(row.authorName),
            'organization': str(row.orgName)
        }
        if author_info not in paper_info['authors']:
            paper_info['authors'].append(author_info)
        
        org_info = {
            'name': str(row.orgName)
        }
        if org_info not in paper_info['organizations']:
            paper_info['organizations'].append(org_info)
        
        if hasattr(row, 'funderName') and row.funderName:
            funder_info = {
                'name': str(row.funderName)
            }
            if funder_info not in paper_info['funders']:
                paper_info['funders'].append(funder_info)
    
    # Obtener topic del paper desde el grafo de similarity
    if similarity_graph:
        topic_query = f"""
        PREFIX schema: <http://schema.org/>
        PREFIX ex: <http://example.org/>
        SELECT ?topicName WHERE {{
            ex:{paper_id} ex:belongsToTopic ?topic .
            ?topic schema:name ?topicName .
        }}
        """
        
        topic_results = execute_sparql_query(similarity_graph, topic_query)
        if topic_results:
            paper_info['topic'] = str(topic_results[0].topicName)
    return paper_info

# Función para obtener similarity score entre dos papers
def get_similarity_score(similarity_graph, paper1_id, paper2_id):
    """Obtiene el similarity score entre dos papers"""
    if not similarity_graph:
        return None
    
    query = f"""
    PREFIX ex: <http://example.org/>
    SELECT ?score WHERE {{
    {{
        ex:{paper1_id} ex:hasSimilarityTo ex:{paper2_id} ;
                    ex:similarityScore ?score .
    }}
    UNION
    {{
        ex:{paper2_id} ex:hasSimilarityTo ex:{paper1_id} ;
                    ex:similarityScore ?score .
    }}
    }}
    """

    results = execute_sparql_query(similarity_graph, query)
    if results:
        return float(str(results[0].score))
    
    return None

# Función principal de la aplicación
def main():
    st.title("Explorador de Publicaciones Académicas")    
    # Cargar datos RDF
    if 'rdf_graph' not in st.session_state:
        with st.spinner('Cargando datos RDF...'):
            st.session_state.rdf_graph = load_custom_rdf()
            st.session_state.similarity_graph = load_similarity_rdf()
            st.session_state.nx_graph = create_networkx_graph(
                st.session_state.rdf_graph            
            )
            st.session_state.similarity_nx_graph = create_similarity_networkx_graph(
                st.session_state.rdf_graph,
                st.session_state.similarity_graph
            )
    
    # Sidebar para navegación
    st.sidebar.title("Navegación")
    section = st.sidebar.selectbox(
        "Selecciona una sección:",
        ["Grafo Interactivo", "Análisis de Papers", "Comparación de Papers", "Consultas SPARQL"]
    )
    
    if section == "Grafo Interactivo":
        st.header("Grafo Interactivo")
        
        # Selector del tipo de grafo
        graph_type = st.radio(
            "Selecciona el tipo de grafo:",
            ["Grafo Personas y Organizaciones", "Grafo de Similaridad y Topics"],
            help="El grafo completo muestra todas las entidades y relaciones. El grafo de similarity solo muestra papers conectados por su similitud."
        )
        
        # Configuración del grafo
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("🎛️ Controles")
            physics_enabled = st.checkbox("Física habilitada", value=True)
            
            if graph_type == "Grafo Personas y Organizaciones":
                st.markdown("**Leyenda:**")
                st.markdown("🔴 Papers")
                st.markdown("🟢 Personas") 
                st.markdown("🟣 Organizaciones")
                
                # Mostrar métricas del grafo completo
                st.subheader("📈 Estadísticas")
                st.metric("Papers", len([n for n in st.session_state.nx_graph.nodes() 
                                       if st.session_state.nx_graph.nodes[n].get('type') == 'ScholarlyArticle']))
                st.metric("Personas", len([n for n in st.session_state.nx_graph.nodes() 
                                        if st.session_state.nx_graph.nodes[n].get('type') == 'Person']))
                st.metric("Organizaciones", len([n for n in st.session_state.nx_graph.nodes() 
                                               if st.session_state.nx_graph.nodes[n].get('type') == 'Organization']))
                st.metric("Conexiones", st.session_state.nx_graph.number_of_edges())
            
            else:  # Grafo de Similarity
                st.markdown("**Leyenda:**")
                st.markdown("🔴 Papers")
                st.markdown("🔵 Topics")
                # Mostrar métricas del grafo de similarity
                st.subheader("📈 Estadísticas de Similarity")
                paper = len([n for n in st.session_state.nx_graph.nodes() 
                                       if st.session_state.nx_graph.nodes[n].get('type') == 'ScholarlyArticle'])
                similarity_graph = st.session_state.similarity_nx_graph
                st.metric("Papers", paper)
                st.metric("Topics", similarity_graph.number_of_nodes() - paper)
                # Calcular estadísticas de similarity scores
                if similarity_graph.number_of_edges() > 0:
                    scores = [data['score'] for _, _, data in similarity_graph.edges(data=True) 
                             if 'score' in data]
                    if scores:
                        st.metric("Score Promedio", f"{sum(scores)/len(scores):.3f}")
                        st.metric("Score Máximo", f"{max(scores):.3f}")
                        st.metric("Score Mínimo", f"{min(scores):.3f}")
        
        with col1:
            # Crear y mostrar el grafo con agraph
            if graph_type == "Grafo Personas y Organizaciones":
                nodes, edges = create_agraph_visualization(st.session_state.nx_graph)
                current_graph = st.session_state.nx_graph
            else:  # Grafo de Similarity
                if st.session_state.similarity_nx_graph.number_of_nodes() > 0:
                    nodes, edges = create_similarity_agraph_visualization(st.session_state.similarity_nx_graph)
                    current_graph = st.session_state.similarity_nx_graph
                else:
                    st.warning("No hay datos de similarity disponibles. Asegúrate de que el archivo similarity_data.rdf esté cargado correctamente.")
                    return
            
            # Configurar agraph dinámicamente
            config = get_agraph_config()
            config.physics = physics_enabled
            
            # Mostrar el grafo
            selected_node = agraph(
                nodes=nodes,
                edges=edges,
                config=config
            )
            
            # Mostrar información del nodo seleccionado
            if selected_node:
                # Obtener información del nodo desde el grafo NetworkX
                if selected_node in current_graph.nodes():
                    node_data = current_graph.nodes[selected_node]
                    node_type = node_data.get('type', 'Desconocido')
                    node_label = node_data.get('label', selected_node)
                    st.subheader(f"{node_label}")
                    st.write(f"**Tipo:** {node_type}")
                    
                    # Mostrar conexiones
                    neighbors = list(current_graph.neighbors(selected_node))
                    if neighbors:
                        if graph_type == "Grafo de Similarity":
                            st.write(f"**Papers Similares ({len(neighbors)}):**")
                            for neighbor in neighbors:
                                neighbor_label = current_graph.nodes[neighbor].get('label', neighbor)
                                # Obtener el score de similarity
                                edge_data = current_graph.get_edge_data(selected_node, neighbor)
                                if edge_data and 'score' in edge_data:
                                    score = edge_data['score']
                                    st.write(f"- {neighbor_label} (**{score*100:.1f}%** similitud)")
                                else:
                                    st.write(f"- {neighbor_label}")
                        else:
                            st.write(f"**Conexiones ({len(neighbors)}):**")
                            for neighbor in neighbors:
                                neighbor_type = current_graph.nodes[neighbor].get('type', 'Desconocido')
                                neighbor_label = current_graph.nodes[neighbor].get('label', neighbor)
                                st.write(f"- {neighbor_label} ({neighbor_type})")
        
        # Tips de uso
        if graph_type == "Grafo Personas y Organizaciones":
            st.info("💡 **Tips de uso:** Haz clic en cualquier nodo para ver sus detalles y conexiones. Puedes arrastrar los nodos para reorganizar el grafo y usar la rueda del mouse para hacer zoom.")
        else:
            st.info("💡 **Tips de uso:** Este grafo muestra solo papers y sus similarity scores. El grosor y color de las líneas indican el nivel de similitud. Haz clic en un paper para ver todos los papers similares y sus porcentajes de similitud.")
    
    elif section == "Análisis de Papers":
        st.header("📚 Análisis Detallado de Papers")
        
        # Obtener lista de papers
        papers_query = """
        PREFIX schema: <http://schema.org/>
        SELECT ?paper ?title WHERE {
            ?paper a schema:ScholarlyArticle ;
                   schema:name ?title .
        }
        """
        
        papers_results = execute_sparql_query(st.session_state.rdf_graph, papers_query)
        
        if papers_results:
            # Crear selectbox con papers
            paper_options = {}
            for row in papers_results:
                paper_id = str(row.paper).split('/')[-1]
                paper_title = str(row.title)
                paper_options[f"{paper_title} ({paper_id})"] = paper_id
            
            selected_paper_display = st.selectbox(
                "Selecciona un paper para analizar:",
                list(paper_options.keys())
            )
            
            if selected_paper_display:
                selected_paper_id = paper_options[selected_paper_display]
                
                # Obtener detalles del paper
                paper_details = get_paper_details(
                    st.session_state.rdf_graph, 
                    st.session_state.similarity_graph, 
                    selected_paper_id
                )
                if paper_details:
                    # Mostrar información del paper
                    st.subheader(f"📄 {paper_details['title']}")
                    # Información básica
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Fecha de publicación", paper_details['date'])
                    with col2:
                        st.metric("Autores", len(paper_details['authors']))
                    with col3:
                        if paper_details['topic']:
                            st.metric("Topic", paper_details['topic'])
                        else:
                            st.metric("Topic", "No disponible")
                    
                    # Abstract
                    st.subheader("📝 Abstract")
                    st.write(paper_details['abstract'])
                    
                    # Sección de autores
                    st.subheader("👥 Autores y Afiliaciones")
                    authors_data = []
                    for author in paper_details['authors']:
                        authors_data.append({
                            'Autor': author['name']
                        })
                    
                    authors_df = pd.DataFrame(authors_data)
                    st.dataframe(authors_df, use_container_width=True, hide_index=True)
                    
                    # Sección de financiadores
                    if paper_details['funders']:
                        st.subheader("💰 Financiadores")
                        for funder in paper_details['funders']:
                            st.write(f"• {funder['name']}")
                    
                    # Análisis de colaboración
                    if len(paper_details['organizations']) > 1:
                        st.subheader("🤝 Análisis de Colaboración")
                        st.metric("Organizaciones colaboradoras", len(paper_details['organizations']))
                else:
                    st.error("No se pudieron obtener los detalles del paper")
        else:
            st.warning("No se encontraron papers en el grafo RDF")
    
    elif section == "Comparación de Papers":
        st.header("🔍 Comparación de Papers")
        st.markdown("Compara la similitud entre dos papers usando los similarity scores")
        
        # Obtener lista de papers
        papers_query = """
        PREFIX schema: <http://schema.org/>
        SELECT ?paper ?title WHERE {
            ?paper a schema:ScholarlyArticle ;
                   schema:name ?title .
        }
        """
        
        papers_results = execute_sparql_query(st.session_state.rdf_graph, papers_query)
        
        if papers_results:
            # Crear opciones de papers
            paper_options = {}
            for row in papers_results:
                paper_id = str(row.paper).split('/')[-1]
                paper_title = str(row.title)
                paper_options[f"{paper_title} ({paper_id})"] = paper_id
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 Paper 1")
                selected_paper1_display = st.selectbox(
                    "Selecciona el primer paper:",
                    list(paper_options.keys()),
                    key="paper1"
                )
                
                if selected_paper1_display:
                    paper1_id = paper_options[selected_paper1_display]
                    paper1_details = get_paper_details(
                        st.session_state.rdf_graph, 
                        st.session_state.similarity_graph, 
                        paper1_id
                    )
                    
                    if paper1_details:
                        st.write(f"**Título:** {paper1_details['title']}")
                        st.write(f"**Fecha:** {paper1_details['date']}")
                        if paper1_details['topic']:
                            st.write(f"**Topic:** {paper1_details['topic']}")
                        st.write(f"**Autores:** {len(paper1_details['authors'])}")
            
            with col2:
                st.subheader("📄 Paper 2")
                # Filtrar para evitar seleccionar el mismo paper
                paper2_options = {k: v for k, v in paper_options.items() 
                                if k != selected_paper1_display}
                
                selected_paper2_display = st.selectbox(
                    "Selecciona el segundo paper:",
                    list(paper2_options.keys()),
                    key="paper2"
                )
                
                if selected_paper2_display:
                    paper2_id = paper2_options[selected_paper2_display]
                    paper2_details = get_paper_details(
                        st.session_state.rdf_graph, 
                        st.session_state.similarity_graph, 
                        paper2_id
                    )
                    
                    if paper2_details:
                        st.write(f"**Título:** {paper2_details['title']}")
                        st.write(f"**Fecha:** {paper2_details['date']}")
                        if paper2_details['topic']:
                            st.write(f"**Topic:** {paper2_details['topic']}")
                        st.write(f"**Autores:** {len(paper2_details['authors'])}")
            
            # Comparación y similarity score
            if selected_paper1_display and selected_paper2_display:
                st.divider()
                st.subheader("📊 Análisis de Similitud")
                
                paper1_id = paper_options[selected_paper1_display]
                paper2_id = paper2_options[selected_paper2_display]
                # Obtener similarity score
                similarity_score = get_similarity_score(
                    st.session_state.similarity_graph, 
                    paper1_id, 
                    paper2_id
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if similarity_score is not None:
                        st.metric("Similarity Score", f"{(abs(similarity_score) * 100):.2f}%")
                        
                        # Interpretación del score
                        if similarity_score >= 0.8:
                            st.success("🟢 Muy similar")
                        elif similarity_score >= 0.6:
                            st.info("🟡 Moderadamente similar")
                        elif similarity_score >= 0.4:
                            st.warning("🟠 Poco similar")
                        else:
                            st.error("🔴 Muy diferente")
                    else:
                        st.metric("Similarity Score", "No disponible")
                        st.info("No se encontró información de similitud entre estos papers")
                
                with col2:
                    # Comparar topics
                    if (paper1_details and paper2_details and 
                        paper1_details['topic'] and paper2_details['topic']):
                        
                        if paper1_details['topic'] == paper2_details['topic']:
                            st.metric(f"Ambos pertenecen a:",f"{paper1_details['topic']}")
                            st.success("✅ Mismo topic")
                        else:
                            st.write(f"Paper 1: **{paper1_details['topic']}**")
                            st.write(f"Paper 2: **{paper2_details['topic']}**")
                            st.warning("❌ Topics diferentes")
                
                with col3:
                    # Comparar años de publicación
                    if paper1_details and paper2_details:
                        year1 = paper1_details['date'][:4]
                        year2 = paper2_details['date'][:4]
                        
                        if year1 != "Unkn" and year2 != "Unkn":
                            year_diff = abs(int(year1) - int(year2))
                        
                            st.metric("Diferencia de años", f"{year_diff} años")
                        
                            if year_diff == 0:
                                st.success("Mismo año")
                            elif year_diff <= 2:
                                st.info("Años cercanos")
                            else:
                                st.warning("Años distantes")
                        else:
                                st.error("Fechas no disponibles para comparación")
                


        else:
            st.warning("No se encontraron papers para comparar")
    
    elif section == "Consultas SPARQL":
        st.header("🔍 Consultas SPARQL Personalizadas")
        st.markdown("Ejecuta consultas SPARQL personalizadas sobre los datos RDF")
        
        # Consultas predefinidas
        st.subheader("Consultas Predefinidas")
        
        predefined_queries = {
            "Todos los papers y sus abstracts": """
PREFIX schema: <http://schema.org/>
SELECT ?paper ?title ?abstract WHERE {
    ?paper a schema:ScholarlyArticle ;
           schema:name ?title ;
           schema:abstract ?abstract .
}
            """,
            "Autores": """
PREFIX schema: <http://schema.org/>
SELECT ?authorName  WHERE {
    ?author a schema:Person ;
            schema:name ?authorName .
}
            """,
            "Papers por año de publicación": """
PREFIX schema: <http://schema.org/>

SELECT ?year (COUNT(?paper) AS ?number_of_papers)
WHERE {
  ?paper a schema:ScholarlyArticle ;
         schema:datePublished ?date .
  BIND(SUBSTR(STR(?date), 1, 4) AS ?year)
}
GROUP BY ?year
ORDER BY ?year
            """,
            "Organizaciones que financian papers": """
PREFIX schema: <http://schema.org/>
SELECT ?orgName (COUNT(?paper) as ?papersFinanciados) WHERE {
    ?paper schema:funder ?org .
    ?org schema:name ?orgName .
}
GROUP BY ?org ?orgName
ORDER BY DESC(?papersFinanciados)
            """,
            "Papers por topic": """
PREFIX schema: <http://schema.org/>
PREFIX ex: <http://example.org/>
SELECT ?topicName (COUNT(?paper) as ?paperCount) WHERE {
    ?paper ex:belongsToTopic ?topic .
    ?topic schema:name ?topicName .
}
GROUP BY ?topic ?topicName
ORDER BY DESC(?paperCount)
            """,
            "Top 10 pares de papers más similares": """
PREFIX ex: <http://example.org/>
PREFIX schema: <http://schema.org/>
SELECT ?paper1Title ?paper2Title ?score WHERE {
    ?paper1 ex:hasSimilarityTo ?paper2 ;
            ex:similarityScore ?score ;
            schema:name ?paper1Title .
    ?paper2 schema:name ?paper2Title .
    FILTER(?score > 0.5)
}
ORDER BY DESC(?score)
LIMIT 10
            """
        }
        
        selected_query = st.selectbox("Selecciona una consulta:", list(predefined_queries.keys()))
        
        if selected_query:
            st.code(predefined_queries[selected_query], language='sparql')
            
            if st.button("Ejecutar Consulta Predefinida"):
                # Decidir qué grafo usar según la consulta
                if "topic" in selected_query.lower() or "similares" in selected_query.lower():
                    graph_to_use = st.session_state.similarity_graph
                    if not graph_to_use or len(graph_to_use) == 0:
                        st.warning("Esta consulta requiere el archivo de similarity que no está disponible.")
                        return
                else:
                    graph_to_use = st.session_state.rdf_graph
                
                results = execute_sparql_query(graph_to_use, predefined_queries[selected_query])
                
                if results:
                    # Convertir resultados a DataFrame
                    rows = []
                    for row in results:
                        row_dict = {}
                        for var in row.labels:
                            row_dict[var] = str(row[var])
                        rows.append(row_dict)
                    
                    df = pd.DataFrame(rows)
                    st.subheader("Resultados:")
                    st.dataframe(df, use_container_width=True)
                    
                    # Mostrar gráfico si es apropiado
                    if any(col in df.columns for col in ['count', 'papersFinanciados', 'paperCount', 'number_of_papers']):
                        count_cols = [col for col in ['count', 'papersFinanciados', 'paperCount', 'number_of_papers'] if col in df.columns]
                        if count_cols:
                            col_name = count_cols[0]
                            x_col = df.columns[0]
                            
                            fig = px.bar(df, x=x_col, y=col_name, 
                                       title=f"Visualización: {selected_query}")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Gráfico especial para similarity scores
                    elif 'score' in df.columns and len(df) > 1:
                        fig = px.histogram(df, x='score', nbins=20,
                                         title="Distribución de Similarity Scores")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Estadísticas de similarity
                        st.subheader("📊 Estadísticas de Similitud")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Score Promedio", f"{df['score'].astype(float).mean():.4f}")
                        with col2:
                            st.metric("Score Máximo", f"{df['score'].astype(float).max():.4f}")
                        with col3:
                            st.metric("Score Mínimo", f"{df['score'].astype(float).min():.4f}")
                        with col4:
                            st.metric("Desviación Estándar", f"{df['score'].astype(float).std():.4f}")
                else:
                    st.info("La consulta no devolvió resultados")
        
        # Consulta personalizada
        st.subheader("Consulta Personalizada")
        
        # Selector de grafo para consultas personalizadas
        graph_choice = st.radio(
            "Selecciona el grafo para la consulta:",
            ["Grafo Principal", "Grafo de Similarity", "Ambos grafos"],
            help="El grafo principal contiene papers, autores y organizaciones. El grafo de similarity contiene topics y similarity scores."
        )
        
        custom_query = st.text_area(
            "Escribe tu consulta SPARQL:",
            height=200,
            placeholder="""PREFIX schema: <http://schema.org/>
PREFIX ex: <http://example.org/>
SELECT ?s ?p ?o WHERE {
    ?s ?p ?o .
    LIMIT 10
}"""
        )
        
        if st.button("Ejecutar Consulta Personalizada"):
            if custom_query.strip():
                results = []
                
                if graph_choice == "Grafo Principal":
                    results = execute_sparql_query(st.session_state.rdf_graph, custom_query)
                elif graph_choice == "Grafo de Similarity":
                    if st.session_state.similarity_graph and len(st.session_state.similarity_graph) > 0:
                        results = execute_sparql_query(st.session_state.similarity_graph, custom_query)
                    else:
                        st.warning("El grafo de similarity no está disponible.")
                else:  # Ambos grafos
                    st.subheader("Resultados del Grafo Principal:")
                    results_main = execute_sparql_query(st.session_state.rdf_graph, custom_query)
                    if results_main:
                        rows_main = []
                        for row in results_main:
                            row_dict = {}
                            for var in row.labels:
                                row_dict[var] = str(row[var])
                            rows_main.append(row_dict)
                        df_main = pd.DataFrame(rows_main)
                        st.dataframe(df_main, use_container_width=True)
                    else:
                        st.info("Sin resultados en el grafo principal")
                    
                    st.subheader("Resultados del Grafo de Similarity:")
                    if st.session_state.similarity_graph and len(st.session_state.similarity_graph) > 0:
                        results_sim = execute_sparql_query(st.session_state.similarity_graph, custom_query)
                        if results_sim:
                            rows_sim = []
                            for row in results_sim:
                                row_dict = {}
                                for var in row.labels:
                                    row_dict[var] = str(row[var])
                                rows_sim.append(row_dict)
                            df_sim = pd.DataFrame(rows_sim)
                            st.dataframe(df_sim, use_container_width=True)
                        else:
                            st.info("Sin resultados en el grafo de similarity")
                    else:
                        st.warning("El grafo de similarity no está disponible.")
                    return
                
                if results:
                    rows = []
                    for row in results:
                        row_dict = {}
                        for var in row.labels:
                            row_dict[var] = str(row[var])
                        rows.append(row_dict)
                    
                    df = pd.DataFrame(rows)
                    st.subheader("Resultados:")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("La consulta no devolvió resultados")
            else:
                st.warning("Por favor, ingresa una consulta SPARQL")
    
if __name__ == "__main__":
    main()
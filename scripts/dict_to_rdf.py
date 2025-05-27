from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, OWL
import json
import requests
import os

def quitar_espacios(cadena):
    return cadena.replace(" ", "_").replace("/", "_").replace(":", "_")

def get_wikidata_uri(nombre):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": nombre,
        "language": "en",
        "format": "json"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if len(data['search']) > 0:
            return 'https:' + data['search'][0]['url']
    except:
        pass
    return None

# Cargar los datos desde tu JSON
json_path = os.path.join("output_files", "Text_Extraction_results.json")
with open(json_path, "r", encoding="utf-8") as f:
    papers = json.load(f)

# Namespaces
SCHEMA = Namespace("http://schema.org/")
EX = Namespace("http://example.org/")

# Crear grafo RDF
g = Graph()
g.bind("schema", SCHEMA)
g.bind("ex", EX)

for file_id, paper_info in papers.items():
    paper_uri = EX[quitar_espacios(file_id.replace(".xml", ""))]

    g.add((paper_uri, RDF.type, SCHEMA.ScholarlyArticle))

    # Título
    if paper_info.get('titulo'):
        g.add((paper_uri, SCHEMA.name, Literal(paper_info['titulo'])))
    else:
        g.add((paper_uri, SCHEMA.name, Literal("No title available")))
    # Abstract (opcional: reconstruido desde palabras si no tienes campo específico)
    if paper_info.get('words'):
        abstract = " ".join(paper_info['words'][:100])
        g.add((paper_uri, SCHEMA.abstract, Literal(abstract)))
    else:
        g.add((paper_uri, SCHEMA.abstract, Literal("No abstract available")))
    # ID
    if paper_info.get('id'):
        tipo, valor = paper_info['id']
        g.add((paper_uri, SCHEMA.identifier, Literal(f"{tipo}:{valor}")))
    else:
        g.add((paper_uri, SCHEMA.identifier, Literal("Unknown ID")))
    # Fecha
    if paper_info.get('fecha'):
        g.add((paper_uri, SCHEMA.datePublished, Literal(paper_info['fecha'])))
    else: 
        g.add((paper_uri, SCHEMA.datePublished, Literal("Unknown")))
    # Autores
    for author_name in paper_info.get('autores', []):
        person_uri = EX["person/" + quitar_espacios(author_name)]
        g.add((person_uri, RDF.type, SCHEMA.Person))
        g.add((person_uri, SCHEMA.name, Literal(author_name)))
        g.add((paper_uri, SCHEMA.author, person_uri))

        # Opcional: añadir enlace a Wikidata
        wikidata_uri = get_wikidata_uri(author_name)
        if wikidata_uri:
            g.add((person_uri, OWL.sameAs, URIRef(wikidata_uri)))

    # Personas mencionadas (NER en acknowledgements)
    for person_name in paper_info.get('personas', []):
        ack_uri = EX["ack_person/" + quitar_espacios(person_name)]
        g.add((ack_uri, RDF.type, SCHEMA.Person))
        g.add((ack_uri, SCHEMA.name, Literal(person_name)))
        g.add((paper_uri, SCHEMA.acknowledges, ack_uri))

    # Organizaciones mencionadas
    for org_name in paper_info.get('organizaciones', []):
        org_uri = EX["org/" + quitar_espacios(org_name)]
        g.add((org_uri, RDF.type, SCHEMA.Organization))
        g.add((org_uri, SCHEMA.name, Literal(org_name)))
        g.add((paper_uri, SCHEMA.funder, org_uri))

        wikidata_uri = get_wikidata_uri(org_name)
        if wikidata_uri:
            g.add((org_uri, OWL.sameAs, URIRef(wikidata_uri)))

    # Palabras (opcionales, como etiquetas o términos clave)
    for word in paper_info.get('words', [])[:10]:  # limitar a 10 por simplicidad
        g.add((paper_uri, EX.hasWord, Literal(word)))

# Guardar el grafo RDF en la misma carpeta que el JSON
output_file = os.path.join("output_files", "knowledge_graph_linked.rdf")
g.serialize(destination=output_file, format="xml")
print(f"Grafo guardado en: {output_file}")
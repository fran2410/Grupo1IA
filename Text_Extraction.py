import os
import re
import xml.etree.ElementTree as ET
import nltk
import torch
import sys
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from lxml import etree
import json

# Redirige la salida estándar al archivo
# sys.stdout = open("NER_results.txt", "w")

# Descargar recursos de NLTK si no están disponibles
nltk.download('punkt')

def split_text(text, max_tokens=256):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_tokens:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def extract_text_from_grobid_xml(xml_path):
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}

    try:
        tree = etree.parse(xml_path)
    except etree.XMLSyntaxError as e:
        print(f"Error parsing {xml_path}: {e}")
        return ""

    root = tree.getroot()
    acknowledgement_div = root.find('.//tei:back/tei:div[@type="acknowledgement"]', namespaces)

    if acknowledgement_div is None:
        return find_acknowledgments(etree.tostring(root, encoding="unicode"))

    def get_full_text(element):
        text = element.text or ""
        for child in element:
            text += get_full_text(child)
            text += child.tail or ""
        return text.strip()

    acknowledgement_texts = []
    for p in acknowledgement_div.findall('.//tei:p', namespaces):
        full_text = get_full_text(p)
        if full_text:
            acknowledgement_texts.append(full_text)

    return ' '.join(acknowledgement_texts)

def find_acknowledgments(xml_string):
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    tree = etree.fromstring(xml_string.encode("utf-8"))
    for div in tree.xpath('//tei:div', namespaces=ns):
        head = div.find('tei:head', namespaces=ns)
        if head is not None and 'acknowledgment' in head.text.lower():
            return ''.join(div.itertext())
    return ""

def extract_words_from_paper(xml_path):
    """
    Extrae todas las palabras del <abstract> y <body> del artículo en un XML de GROBID.
    
    Args:
        xml_path (str): Ruta al archivo XML
    
    Returns:
        list: Lista de palabras (en minúsculas) encontradas en abstract y body
    """
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}

    try:
        tree = etree.parse(xml_path)
    except etree.XMLSyntaxError as e:
        print(f"Error parsing {xml_path}: {e}")
        return []

    root = tree.getroot()
    
    # Buscar las secciones abstract y body
    abstract_el = root.find('.//tei:abstract', namespaces)
    body_el = root.find('.//tei:body', namespaces)

    # Función para extraer todo el texto plano de un elemento
    def extract_text(el):
        if el is None:
            return ""
        return ' '.join(el.itertext())

    full_text = extract_text(abstract_el) + ' ' + extract_text(body_el)
    
    # Limpiar texto y dividir en palabras
    words = re.findall(r'\b\w+\b', full_text.lower())

    return words

def extract_metadata(xml_path):
    """
    Extrae título, DOI, fecha de publicación y autores del XML.
    """
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
    metadata = {
        'titulo': None,
        'doi': None,
        'fecha': None,
        'autores': []
    }

    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()

        # Título
        title_el = root.find('.//tei:titleStmt/tei:title', namespaces)
        if title_el is not None:
            metadata['titulo'] = title_el.text

        # id (DOI preferido, sino arXiv, MD5, etc.)
        idno_el = None
        for idno in root.findall('.//tei:idno', namespaces):
            tipo = idno.get('type', '').upper()
            valor = idno.text
            if tipo and valor:
                if tipo == 'DOI':
                    metadata['id'] = (tipo, valor)
                    break
                elif tipo == 'ARXIV':
                    idno_el = (tipo, valor)
                elif not idno_el:
                    idno_el = (tipo, valor)
        # Si no había DOI, usa el primero disponible
        if 'id' not in metadata and idno_el:
            metadata['id'] = idno_el


        # Fecha (atributo 'when' del <date type="published">)
        date_el = root.find('.//tei:publicationStmt/tei:date[@type="published"]', namespaces)
        if date_el is not None and 'when' in date_el.attrib:
            metadata['fecha'] = date_el.attrib['when']

        # Autores
        for author_el in root.findall('.//tei:sourceDesc//tei:author', namespaces):
            persName = author_el.find('.//tei:persName', namespaces)
            if persName is not None:
                name_parts = []
                for part in ['tei:forename', 'tei:surname']:
                    el = persName.find(part, namespaces)
                    if el is not None:
                        name_parts.append(el.text)
                full_name = ' '.join(name_parts).strip()
                if full_name:
                    metadata['autores'].append(full_name)

    except Exception as e:
        print(f" - Error extrayendo metadatos: {e}")

    return metadata

# Configuración de modelos
MODELS = {
    # "UnBIAS":        "newsmediabias/UnBIAS-Named-Entity-Recognition",
    # "ConfliBERT":    "eventdata-utd/conflibert-named-entity-recognition", 
    # "AventIQ":       "AventIQ-AI/bert-named-entity-recognition",
    # # Modelos basados en SciBERT fine‑tuned para NER científico
    # "SciBERT-JNLPBA":       "siddharthtumre/scibert-finetuned-ner",              
    # "SciBERT-JNLPBA-cased": "fran-martinez/scibert_scivocab_cased_ner_jnlpba",   
    # "PCSciBERT":            "jmzk96/PCSciBERT_uncased",                           
    # # Modelos genéricos de alto rendimiento en CoNLL‑2003
    # "BERT-CoNLL-Base":      "dslim/bert-base-NER",                                
    # "BERT-CoNLL-Cased":     "kamalkraj/bert-base-cased-ner-conll2003",            
    # "BERT-CoNLL-Large":     "dbmdz/bert-large-cased-finetuned-conll03-english",   
    "Jean-Baptiste" : "Jean-Baptiste/roberta-large-ner-english"
}


def initialize_models():
    """Inicializa pipelines de NER para cada modelo"""
    models = {}
    for model_name, model_path in MODELS.items():
        try:
            models[model_name] = pipeline(
                "ner",
                model=model_path,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1  # Usar GPU si está disponible
            )
            print(f"Modelo {model_name} cargado correctamente")
        except Exception as e:
            print(f"Error cargando modelo {model_name}: {str(e)}")
            models[model_name] = None
    return models

def process_xml_directory(xml_dir):
    ner_pipelines = initialize_models()
    output_data = {}

    for filename in os.listdir(xml_dir):
        if not filename.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, filename)
        print(f"\nProcesando: {filename}")

        try:
            full_text = extract_text_from_grobid_xml(xml_path)
            if not full_text:
                print(" - No se pudo extraer texto")
                continue

            text_chunks = split_text(full_text)
            models = {model: {'personas': set(), 'organizaciones': set()} for model in MODELS}

            for chunk in text_chunks:
                for model_name, ner_pipeline in ner_pipelines.items():
                    if ner_pipeline is None:
                        continue
                    try:
                        entities = ner_pipeline(chunk)
                        for entity in entities:
                            text = entity['word'].strip()
                            label = entity['entity_group']
                            if label in ['PERSON', 'PER', 'Person']:
                                models[model_name]['personas'].add(text)
                            elif label in ['ORGANIZATION', 'ORG', 'Organisation']:
                                models[model_name]['organizaciones'].add(text)
                    except Exception as e:
                        print(f" - Error con {model_name}: {str(e)}")
                        continue

            # Extraer palabras
            words = extract_words_from_paper(xml_path)
            metadata = extract_metadata(xml_path)

            safe_filename = filename.replace(" ", "_")
            output_data[safe_filename] = {
                'titulo': metadata['titulo'],
                'id': metadata.get('id'),
                'fecha': metadata['fecha'],
                'autores': metadata['autores'],
                'personas': sorted(list(models[model_name]['personas'])),
                'organizaciones': sorted(list(models[model_name]['organizaciones'])),
                'words': words
            }


        except Exception as e:
            print(f"Error procesando {filename}: {str(e)}")

    # Guardar a JSON
    with open("Text_Extraction_results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py <directorio_xml>")
        sys.exit(1)

    xml_directory = sys.argv[1]
    process_xml_directory(xml_directory)
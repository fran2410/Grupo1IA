import os
import xml.etree.ElementTree as ET
import nltk
import torch
import sys
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from lxml import etree


# Redirige la salida estándar al archivo
sys.stdout = open("NER_results.txt", "w")

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

# Configuración de modelos
MODELS = {
    "UnBIAS":        "newsmediabias/UnBIAS-Named-Entity-Recognition",
    "ConfliBERT":    "eventdata-utd/conflibert-named-entity-recognition", 
    "AventIQ":       "AventIQ-AI/bert-named-entity-recognition",
    # Modelos basados en SciBERT fine‑tuned para NER científico
    "SciBERT-JNLPBA":       "siddharthtumre/scibert-finetuned-ner",              
    "SciBERT-JNLPBA-cased": "fran-martinez/scibert_scivocab_cased_ner_jnlpba",   
    "PCSciBERT":            "jmzk96/PCSciBERT_uncased",                           
    # Modelos genéricos de alto rendimiento en CoNLL‑2003
    "BERT-CoNLL-Base":      "dslim/bert-base-NER",                                
    "BERT-CoNLL-Cased":     "kamalkraj/bert-base-cased-ner-conll2003",            
    "BERT-CoNLL-Large":     "dbmdz/bert-large-cased-finetuned-conll03-english",   
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
    """Procesa todos los archivos XML en un directorio"""
    ner_pipelines = initialize_models()
    
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
            print("Acknowledgments section : \n'", full_text, "'")

            # Dividir el texto en chunks
            text_chunks = split_text(full_text)
            models = {}
            for model in MODELS:
                models[model] = {
                    'nombres': set(),
                    'organizaciones': set()
                }
            # Procesar cada chunk con barra de progreso
            for chunk in text_chunks:
                for model_name, ner_pipeline in ner_pipelines.items():
                    if ner_pipeline is None:
                        continue
                    try:
                        # Procesar sin parámetros no soportados
                        entities = ner_pipeline(chunk)
                        
                        for entity in entities:
                            text = entity['word'].strip()
                            label = entity['entity_group']
                            
                            if label in ['PERSON', 'PER','Person']:
                                models[model_name]['nombres'].add(text)
                            elif label in ['ORGANIZATION', 'ORG','Organisation']:
                                models[model_name]['organizaciones'].add(text)
                                
                    except Exception as e:
                        print(f" - Error con {model_name}: {str(e)}")
                        continue
                    
            # Convertir a listas y mostrar resultados
            print("\n-------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            print(f"\nPaper : {filename}")
            for modelname in MODELS:
                print(f"\nModelo: {modelname}")
                print("\nResultados:")
                print(f"Nombres ({len(models[model_name]['nombres'])}):", models[model_name]['nombres'])
                print(f"Organizaciones ({len(models[model_name]['organizaciones'])}):", models[model_name]['organizaciones'])

        except Exception as e:
            print(f"Error procesando {filename}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py <directorio_xml>")
        sys.exit(1)
        
    xml_directory = sys.argv[1]
    process_xml_directory(xml_directory)
    sys.stdout.close()
    sys.stdout = sys.__stdout__
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

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

INFO_A_MANO = {
    "2405.17044v3.xml": {
        "autores_articulo": ["Xuemei Gu", "Mario Krenn"],
        "personas": [
            "Xuemei Gu"
        ],
        "organizaciones": [
            "OpenAlex",
            "arXiv",
            "bioRxiv",
            "medRxiv",
            "chemRxiv",
            "Alexander von Humboldt Foundation"
        ]
    },
    "2206.02983v2.xml": {
        "autores_articulo": ["Jabir Al Nahian", "Abu Kaisar", "Mohammad Masum"],
        "personas": [],
        "organizaciones": [
            "Computational Intelligence LAB"
        ]
    },
    "A Review of Machine Learning and Deep Learning Applications.pdf.tei.xml": {
        "autores_articulo": ["Pramila P Shinde", "Seema Shah", "Mukesh Patel"],
        "personas": [],
        "organizaciones": []
    },
    "Artificial Intelligence in Education.pdf.tei.xml": {
        "autores_articulo": ["Lijia Chen", "Pingping Chen", "Zhijian Lin"],
        "personas": [],
        "organizaciones": [
            "Humanities and Social Science Planning Funds of Fujian Province",
            "Educational Commission of Fujian Province"
        ]
    },
    "Breast cancer detection through attention based feature integration model.xml": {
        "autores_articulo": ["Sharada Gupta", "Murundi N Eshwarappa"],
        "personas": [],
        "organizaciones": []
    }
}

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
            if current_chunk:
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
    if acknowledgement_div is not None:
        def get_full_text_recursive(element):
            text_content = ""
            if element.text:
                text_content += element.text.strip() + " "
            for child in element:
                text_content += get_full_text_recursive(child)
                if child.tail:
                    text_content += child.tail.strip() + " "
            return text_content
        full_ack_text = get_full_text_recursive(acknowledgement_div).strip()
        full_ack_text = re.sub(r'\s+', ' ', full_ack_text)
        return full_ack_text
    
    for div in root.xpath('//tei:div', namespaces=namespaces):
        head_element = div.find('tei:head', namespaces=namespaces)
        if head_element is not None and head_element.text is not None and \
           ('acknowledgment' in head_element.text.lower() or \
            'acknowledgement' in head_element.text.lower() or \
            'funding' in head_element.text.lower()):
            
            p_texts = [" ".join(p.itertext()).strip() for p in div.findall('.//tei:p', namespaces)]
            combined_text = " ".join(filter(None, p_texts))
            combined_text = re.sub(r'\s+', ' ', combined_text).strip() 
            if combined_text:
                 return combined_text
    return ""

def extract_words_from_paper(xml_path):
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
    try:
        tree = etree.parse(xml_path)
    except etree.XMLSyntaxError as e:
        print(f"Error parsing {xml_path}: {e}")
        return []
    root = tree.getroot()
    abstract_el = root.find('.//tei:abstract', namespaces)
    body_el = root.find('.//tei:body', namespaces)
    def extract_text(el):
        if el is None:
            return ""
        return ' '.join(el.itertext())
    full_text = extract_text(abstract_el) + ' ' + extract_text(body_el)
    words = re.findall(r'\b\w+\b', full_text.lower())
    return words

def extract_metadata(xml_path):
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
    metadata = { 
        'titulo': None,
        'id': None,
        'fecha': None,
        'autores_xml': [] 
    }
    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()

        title_el = root.find('.//tei:titleStmt/tei:title[@level="a"][@type="main"]', namespaces)
        if title_el is None:
             title_el = root.find('.//tei:titleStmt/tei:title', namespaces)
        if title_el is not None:
            title_text_list = list(title_el.itertext())
            full_title_text = "".join(title_text_list).strip()
            metadata['titulo'] = re.sub(r'\s+', ' ', full_title_text)

        id_val = None
        doi_el = root.find('.//tei:idno[@type="DOI"]', namespaces)
        if doi_el is not None and doi_el.text:
            id_val = ("DOI", doi_el.text.strip())
        else: 
            for idno_el_iter in root.findall('.//tei:idno', namespaces):
                tipo = idno_el_iter.get('type', 'UNKNOWN').upper()
                valor = idno_el_iter.text
                if valor: 
                    valor = valor.strip()
                    if tipo == "ARXIV": 
                         id_val = (tipo, valor)
                         break
                    if not id_val: 
                         id_val = (tipo, valor)
        metadata['id'] = id_val

        date_el = root.find('.//tei:publicationStmt/tei:date[@type="published"]', namespaces)
        if date_el is None: 
            date_el = root.find('.//tei:publicationStmt/tei:date', namespaces)
        if date_el is not None:
            if 'when' in date_el.attrib:
                metadata['fecha'] = date_el.attrib['when']
            elif date_el.text: 
                 metadata['fecha'] = date_el.text.strip()
        
        for author_el in root.findall('.//tei:analytic//tei:author', namespaces): 
            persName = author_el.find('.//tei:persName', namespaces)
            if persName is not None:
                name_parts = []
                for fn_el in persName.findall('tei:forename', namespaces):
                    if fn_el.text:
                        name_parts.append(fn_el.text.strip())
                sn_el = persName.find('tei:surname', namespaces)
                if sn_el is not None and sn_el.text:
                    name_parts.append(sn_el.text.strip())
                
                full_name = ' '.join(filter(None, name_parts)) 
                if full_name and full_name not in metadata['autores_xml']:
                    metadata['autores_xml'].append(full_name)
                    
    except Exception as e:
        print(f" - Error extrayendo metadatos para {xml_path}: {type(e).__name__} - {e}")
    
    return metadata

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
    models_pipelines = {} 
    if not MODELS: 
        print("No hay modelos definidos en la constante MODELS.")
        return models_pipelines
    for model_name, model_path in MODELS.items():
        try:
            print(f"Intentando cargar el modelo: {model_name} desde {model_path}")
            device_to_use = 0 if torch.cuda.is_available() else -1
            if device_to_use == 0:
                print("CUDA disponible, intentando usar GPU.")
            else:
                print("CUDA no disponible, usando CPU.")
            models_pipelines[model_name] = pipeline(
                "ner",
                model=model_path,
                tokenizer=model_path, 
                aggregation_strategy="simple",
                device=device_to_use 
            )
            print(f"Modelo {model_name} cargado correctamente en dispositivo {'GPU' if device_to_use == 0 else 'CPU'}.")
        except Exception as e:
            print(f"Error detallado cargando modelo {model_name}: {type(e).__name__} - {str(e)}")
            models_pipelines[model_name] = None 
    return models_pipelines

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1_score": f1_score, "tp": tp, "fp": fp, "fn": fn}

def compare_and_evaluate_models(all_predictions_data, info_a_mano_data, active_models_names):
    evaluation_results = {}
    for filename, data_for_file in all_predictions_data.items():
        if filename not in info_a_mano_data:
            print(f"Advertencia: No hay datos de 'info_a_mano' para {filename}. Saltando evaluación.")
            continue
        
        if 'predictions_by_model' not in data_for_file or not data_for_file['predictions_by_model']:
            print(f"Advertencia: No hay 'predictions_by_model' o están vacías para {filename}. Saltando evaluación para este archivo.")
            continue
            
        evaluation_results[filename] = {}
        
        true_autores_list = info_a_mano_data[filename].get('autores_articulo', [])
        true_personas_gt_list = info_a_mano_data[filename].get('personas', [])
        true_organizaciones_list = info_a_mano_data[filename].get('organizaciones', [])

        true_autores_set = set(a.strip().lower() for a in true_autores_list)
        true_personas_gt_set = set(p.strip().lower() for p in true_personas_gt_list) 
        true_organizaciones_set = set(o.strip().lower() for o in true_organizaciones_list)

        predictions_all_models_for_file = data_for_file['predictions_by_model']

        for model_name in active_models_names:
            if model_name not in predictions_all_models_for_file:
                print(f"Advertencia: El modelo activo {model_name} no tiene predicciones en la estructura de {filename}.")
                continue
            
            model_specific_preds = predictions_all_models_for_file[model_name]
            evaluation_results[filename][model_name] = {}

            pred_personas_raw_set_for_authors = model_specific_preds.get('personas', set()) 
            pred_personas_normalized_set_for_authors = {p.strip().lower() for p in pred_personas_raw_set_for_authors}
            
            tp_aut = len(pred_personas_normalized_set_for_authors.intersection(true_autores_set))
            fp_aut = len(pred_personas_normalized_set_for_authors.difference(true_autores_set)) 
            fn_aut = len(true_autores_set.difference(pred_personas_normalized_set_for_authors)) 
            evaluation_results[filename][model_name]['autores_articulo'] = calculate_metrics(tp_aut, fp_aut, fn_aut)

            pred_personas_normalized_set_general = pred_personas_normalized_set_for_authors
            
            tp_p_gen = len(pred_personas_normalized_set_general.intersection(true_personas_gt_set))
            fp_p_gen = len(pred_personas_normalized_set_general.difference(true_personas_gt_set))
            fn_p_gen = len(true_personas_gt_set.difference(pred_personas_normalized_set_general))
            evaluation_results[filename][model_name]['personas_generales'] = calculate_metrics(tp_p_gen, fp_p_gen, fn_p_gen)

            pred_organizaciones_raw_set = model_specific_preds.get('organizaciones', set())
            pred_organizaciones_normalized_set = {o.strip().lower() for o in pred_organizaciones_raw_set}

            tp_org = len(pred_organizaciones_normalized_set.intersection(true_organizaciones_set))
            fp_org = len(pred_organizaciones_normalized_set.difference(true_organizaciones_set))
            fn_org = len(true_organizaciones_set.difference(pred_organizaciones_normalized_set))
            evaluation_results[filename][model_name]['organizaciones'] = calculate_metrics(tp_org, fp_org, fn_org)
            
    return evaluation_results

def print_evaluation_results(evaluation_results):
    print("\n--- RESULTADOS DE EVALUACIÓN ---")
    for filename, model_data in evaluation_results.items():
        print(f"\nArchivo: {filename}")
        for model_name, entity_metrics in model_data.items():
            print(f"  Modelo: {model_name}")
            if 'autores_articulo' in entity_metrics:
                metrics_aut = entity_metrics['autores_articulo']
                print(f"    Autores Artículo: P: {metrics_aut['precision']:.3f}, R: {metrics_aut['recall']:.3f}, F1: {metrics_aut['f1_score']:.3f} (TP:{metrics_aut['tp']}, FP:{metrics_aut['fp']}, FN:{metrics_aut['fn']})")
            if 'personas_generales' in entity_metrics:
                metrics_p = entity_metrics['personas_generales']
                print(f"    Personas (Gen.): P: {metrics_p['precision']:.3f}, R: {metrics_p['recall']:.3f}, F1: {metrics_p['f1_score']:.3f} (TP:{metrics_p['tp']}, FP:{metrics_p['fp']}, FN:{metrics_p['fn']})")
            if 'organizaciones' in entity_metrics:
                metrics_o = entity_metrics['organizaciones']
                print(f"    Organizaciones: P: {metrics_o['precision']:.3f}, R: {metrics_o['recall']:.3f}, F1: {metrics_o['f1_score']:.3f} (TP:{metrics_o['tp']}, FP:{metrics_o['fp']}, FN:{metrics_o['fn']})")

def process_xml_directory(xml_dir, perform_evaluation):
    ner_pipelines_dict = initialize_models()
    active_model_names = [name for name, pipeline_instance in ner_pipelines_dict.items() if pipeline_instance is not None]

    if not active_model_names: 
        print("No se cargaron modelos activos. Terminando proceso para el directorio.")
        return

    output_data_for_json = {} 
    all_predictions_for_eval = {} if perform_evaluation else None

    for filename in os.listdir(xml_dir):
        if not filename.endswith(".xml"):
            continue
        xml_path = os.path.join(xml_dir, filename)
        print(f"\nProcesando: {filename}")
        
        current_file_model_predictions = {model_name: {'personas': set(), 'organizaciones': set()} for model_name in active_model_names}
        metadata = {'titulo': None, 'id': None, 'fecha': None, 'autores_xml': []} 
        words = []
        full_ack_text_content = ""

        try:
            full_ack_text_content = extract_text_from_grobid_xml(xml_path)
            if not full_ack_text_content:
                pass
            
            text_chunks = split_text(full_ack_text_content) if full_ack_text_content else []

            if text_chunks: 
                for chunk_idx, chunk in enumerate(text_chunks):
                    if not chunk.strip(): continue
                    for model_name in active_model_names: 
                        ner_pipeline_instance = ner_pipelines_dict.get(model_name) 
                        if ner_pipeline_instance is None: 
                            continue
                        try:
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", UserWarning)
                                warnings.simplefilter("ignore", FutureWarning) 
                                entities = ner_pipeline_instance(chunk)
                            for entity in entities:
                                text = entity['word'].strip()
                                label = entity['entity_group']
                                if label in ['PERSON', 'PER', 'Person', 'I-PER', 'B-PER']:
                                    current_file_model_predictions[model_name]['personas'].add(text)
                                elif label in ['ORGANIZATION', 'ORG', 'Organisation', 'I-ORG', 'B-ORG', 'MISC']:
                                    current_file_model_predictions[model_name]['organizaciones'].add(text)
                        except Exception as e_ner:
                            print(f" - Error NER con {model_name} en chunk: '{chunk[:50]}...': {type(e_ner).__name__} - {str(e_ner)[:100]}...") 
                            continue
            
            metadata = extract_metadata(xml_path)
            words = extract_words_from_paper(xml_path)

            output_data_for_json[filename] = {
                'titulo': metadata.get('titulo'), 
                'id': metadata.get('id'),
                'fecha': metadata.get('fecha'),
                'autores_xml': metadata.get('autores_xml', []), 
                'predictions_by_model': {
                    model: {
                        'personas': sorted(list(preds['personas'])),
                        'organizaciones': sorted(list(preds['organizaciones']))
                    } for model, preds in current_file_model_predictions.items()
                },
                'words': words,
                'texto_agradecimientos_extraido': full_ack_text_content if full_ack_text_content else "N/A"
            }
            
            if perform_evaluation and all_predictions_for_eval is not None:
                all_predictions_for_eval[filename] = {
                    'predictions_by_model': current_file_model_predictions 
                }
        
        except Exception as e_file_processing: 
            print(f"Error general procesando el archivo {filename}: {type(e_file_processing).__name__} - {str(e_file_processing)}")
            if filename not in output_data_for_json:
                 output_data_for_json[filename] = {
                    'titulo': metadata.get('titulo'), 
                    'id': metadata.get('id'),
                    'fecha': metadata.get('fecha'),
                    'autores_xml': metadata.get('autores_xml', []),
                    'predictions_by_model': { 
                        model_n: {'personas': [], 'organizaciones': []} for model_n in active_model_names
                    },
                    'words': words,
                    'texto_agradecimientos_extraido': full_ack_text_content if full_ack_text_content else "N/A",
                    'error_procesamiento': str(e_file_processing)
                }

    with open("Text_Extraction_results_detailed.json", "w", encoding="utf-8") as f:
        json.dump(output_data_for_json, f, indent=2, ensure_ascii=False)
    print("\nResultados detallados guardados en Text_Extraction_results_detailed.json")

    if perform_evaluation and all_predictions_for_eval:
        if not all_predictions_for_eval: 
            print("No se recolectaron datos para la evaluación (all_predictions_for_eval está vacío).")
        else:
            evaluation_metrics = compare_and_evaluate_models(all_predictions_for_eval, INFO_A_MANO, active_model_names)
            if evaluation_metrics: 
                print_evaluation_results(evaluation_metrics)
                with open("Evaluation_Metrics.json", "w", encoding="utf-8") as f_eval:
                    json.dump(evaluation_metrics, f_eval, indent=2, ensure_ascii=False)
                print("Métricas de evaluación guardadas en Evaluation_Metrics.json")
            else:
                print("La evaluación no produjo métricas (posiblemente no hay archivos en INFO_A_MANO que coincidan con los procesados o no hubo predicciones válidas).")

    elif perform_evaluation: 
         print("La evaluación fue solicitada (-e), pero no se generaron/recolectaron predicciones para evaluar.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python tu_script.py <directorio_xml> [-e]")
        sys.exit(1)

    xml_directory = sys.argv[1]
    perform_evaluation_flag = False

    if len(sys.argv) > 2 and sys.argv[2] == "-e":
        perform_evaluation_flag = True
        print("Evaluación de modelos ACTIVADA.")
    else:
        print("Evaluación de modelos DESACTIVADA. Para activar, use la flag -e.")

    process_xml_directory(xml_directory, perform_evaluation_flag)
    print("Proceso completado.")
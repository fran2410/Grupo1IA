import os
import glob
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer

# Tamaño máximo de tokens para cada fragmento (incluye [CLS] y [SEP])
MAX_TOKENS = 512
# Estrategia de stride para solapamiento (evita perder entidades en los límites)
STRIDE = 50


def extract_text(xml_path):
    """
    Extrae y devuelve el texto completo de un archivo XML usando un parser XML.
    """
    with open(xml_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, features="xml")
    return soup.get_text(separator=' ', strip=True)


def get_chunks(text: str, tokenizer, max_length=MAX_TOKENS, stride=STRIDE):
    """
    Genera fragmentos de texto solapados que no superan max_length tokens.
    """
    # Encode sin special tokens para luego añadirlos en pipeline
    tokenized = tokenizer(text, return_overflowing_tokens=True,
                          max_length=max_length,
                          stride=stride,
                          return_offsets_mapping=False)
    for input_ids in tokenized['input_ids']:
        yield tokenizer.decode(input_ids, clean_up_tokenization_spaces=True)


def extract_entities(text: str, ner_pipeline, tokenizer):
    """
    Aplica el modelo NER al texto en fragmentos.
    Devuelve listas de nombres de personas y organizaciones sin duplicados.
    """
    all_results = []
    for chunk in get_chunks(text, tokenizer):
        entities = ner_pipeline(chunk)
        all_results.extend(entities)

    names = []
    orgs = []
    for ent in all_results:
        label = ent.get('entity_group', ent.get('entity'))
        if label in ['PER', 'Person', 'PERSON']:
            names.append(ent['word'])
        elif label in ['ORG', 'Organization', 'ORG']:
            orgs.append(ent['word'])

    # Quitar duplicados manteniendo orden
    names = list(dict.fromkeys(names))
    orgs = list(dict.fromkeys(orgs))
    return names, orgs


def main():
    modelos = {
        "UnBIAS": "newsmediabias/UnBIAS-Named-Entity-Recognition",
        "ConfliBERT": "eventdata-utd/conflibert-named-entity-recognition",
        "AventIQ": "AventIQ-AI/bert-named-entity-recognition"
    }

    pipelines = {}
    tokenizers = {}

    # Cargar todos los modelos y tokenizers
    for name, model_id in modelos.items():
        print(f"Cargando modelo: {name}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        ner_pipeline = pipeline(
            "ner",
            model=model_id,
            tokenizer=tokenizer,
            aggregation_strategy="simple"
        )
        pipelines[name] = ner_pipeline
        tokenizers[name] = tokenizer

    archivos = sorted(glob.glob("*.xml"))
    if len(archivos) != 10:
        print(f"Advertencia: Se encontraron {len(archivos)} archivos XML (se esperaban 10).")

    resultados = {}
    for path in archivos:
        paper_id = os.path.splitext(os.path.basename(path))[0]
        print(f"Procesando paper: {paper_id}")
        text = extract_text(path)
        resultados[paper_id] = {}

        for name, ner_pipe in pipelines.items():
            print(f"  Aplicando modelo: {name}")
            tokenizer = tokenizers[name]
            names, orgs = extract_entities(text, ner_pipe, tokenizer)
            resultados[paper_id][name] = {"names": names, "orgs": orgs}

    # Mostrar resumen
    for paper, data in resultados.items():
        print(f"\nResumen para {paper}:")
        for model_name, ents in data.items():
            print(f"  Modelo {model_name} -> Personas: {len(ents['names'])}, Organizaciones: {len(ents['orgs'])}")
            print(f"    Names: {ents['names']}")
            print(f"    Orgs: {ents['orgs']}")


if __name__ == "__main__":
    main()
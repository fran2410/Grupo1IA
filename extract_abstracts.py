import os
from lxml import etree

def extract_all_abstracts(xml_folder):
    abstracts = {}
    namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}

    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            filepath = os.path.join(xml_folder, filename)
            try:
                with open(filepath, 'rb') as f:
                    tree = etree.parse(f)

                    # Encuentra el <p> dentro del abstract, usando namespace
                    abstract_nodes = tree.xpath('//tei:abstract//tei:p', namespaces=namespace)
                    if abstract_nodes:
                        abstract_text = ' '.join([p.text for p in abstract_nodes if p.text])
                        print(f"[✓] Abstract found in {filename}")
                        abstracts[filename] = abstract_text.strip()
                    else:
                        print(f"[✗] No abstract found in {filename}")

            except Exception as e:
                print(f"[ERROR] Failed to parse {filename}: {e}")

    return abstracts

import os
import sys
import xml.etree.ElementTree as ET

# This script process and give a list with all the links of the xml files while ignoring references.
# It can process a sigle file or all the files in a folder.

def extract_links(xml_file):
    '''
    :param: path to the xml file
    :return: list with the links in the xml file
    ''' 
    tree = ET.parse(xml_file)
    root = tree.getroot()
    links = []
    inside_references = False  

    for elem in root.iter():
        # check if we are inside the references section
        if elem.tag == "{http://www.tei-c.org/ns/1.0}div" and elem.get("type") == "references":
            inside_references = True  
        elif inside_references and elem.tag == "{http://www.tei-c.org/ns/1.0}/div":
            inside_references = False  

        if not inside_references:  
            target = elem.get("target")
            if target and should_include_link(target):
                links.append(clean_link(target))
    return list(set(links))  # Remove duplicates

def should_include_link(link):
    # Exclude links to GROBID and TEI-C
    excluded_domains = ['grobid', 'tei-c']
    if any(domain in link.lower() for domain in excluded_domains):
        return False
        
    if link.startswith('#'):
        return False
        
    if not link or not any(prefix in link.lower() for prefix in ['http', 'www', '://']):
        return False
        
    return True

def clean_link(link):
    while link and link[-1] in '.,':
        link = link[:-1]
    return link

def process_files(path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    linkList = []
    if os.path.isfile(path):
        links = extract_links(path)
        linkList.append(f"\nLinks in {path}:")
        for link in sorted(links):
            linkList.append(link)
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith(".xml"):
                filepath = os.path.join(path, filename)
                links = extract_links(filepath)
                linkList.append(f"\nLinks in {filename}:")
                for link in sorted(links):
                    linkList.append(link)
    else:
        print("Invalid path provided.")
    output_path = os.path.join(output_dir, "links.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in linkList:
            f.write(item + "\n")
    print(f"Links saved in {output_path}")
    
if __name__ == "__main__":
# Check ih the user provided enough arguments
    if len(sys.argv) < 2:
        print("Use: python list.py <folder_with_xmls> <output_folder>")
        sys.exit(1)

    path = sys.argv[1]    
    output_dir = sys.argv[2]
    process_files(path, output_dir)
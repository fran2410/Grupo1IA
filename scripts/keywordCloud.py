import os
import sys
import xml.etree.ElementTree as ET
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# This script process and give the keywords cloud of the abstracts of the xml files.
# It can process a sigle file or all the files in a folder.


def extract_abstract(xml_file):
    '''
    :param: path to the xml file
    :return: string with the abstract of the xml file
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()
    abstract = ""
    for elem in root.iter("{http://www.tei-c.org/ns/1.0}abstract"):
        for p in elem.iter("{http://www.tei-c.org/ns/1.0}p"):
            abstract += p.text + " "
    return abstract

def generate_cloud(text, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    output_path = os.path.join(output_dir, "keywordCloud.jpg")
    plt.savefig(output_path, format='jpeg', dpi=300, bbox_inches='tight')

    print(f"Keyword Cloud saved in {output_path}")

def process(path, output_dir):
    if os.path.isfile(path):
        abstract = extract_abstract(path)
        generate_cloud(abstract, output_dir)
    elif os.path.isdir(path):
        all_abstracts = ""
        for filename in os.listdir(path):
            if filename.endswith(".xml"):
                filepath = os.path.join(path, filename)
                all_abstracts += extract_abstract(filepath)
        generate_cloud(all_abstracts, output_dir)
    else:
        print("Invalid path provided.")

if __name__ == "__main__":
    # Check ih the user provided enough arguments
    if len(sys.argv) < 3:
        print("Use: python keywordCloud.py <folder_with_xmls> <output_folder>")
        sys.exit(1)

    path = sys.argv[1]    
    output_dir = sys.argv[2]
    process(path, output_dir)
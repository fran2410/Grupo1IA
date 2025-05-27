import os
import sys
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# This script process the figures of the xml files and gives a chart whitn the number of it.
# It can process a sigle file or all the files in a folder.

titulos = []

def count_figures(xml_file):
    '''
    :param: path to the xml file
    :return: integer with the number of figures in the xml file
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()
    figure_count = 0
    for elem in root.iter("{http://www.tei-c.org/ns/1.0}figure"):
        # print(elem, xml_file)
        figure_count += 1
    # print(figure_count, xml_file)
    return figure_count

def generate_chart(figure_counts, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    bars = plt.bar(range(len(figure_counts)), figure_counts, tick_label=[str(i+1) for i in range(len(figure_counts))])
    plt.xlabel('Documents')
    plt.ylabel('Number of Figures')
    plt.title('Number of Figures per Document')
    cleaned_titles = [tit.replace('.pdf.tei.xml', '') for tit in titulos]
    legend_labels = [f"{i+1}. {cleaned_titles[i]}" for i in range(len(cleaned_titles))]
    plt.legend(bars, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
    
    output_path = os.path.join(output_dir, "charts.jpg")
    plt.savefig(output_path, format='jpeg', dpi=300, bbox_inches='tight')

    print(f"Chart saved in {output_path}")


def process_files(path, output_dir):
    titulos = []
    figure_counts = []
    if os.path.isfile(path):
        titulos.append(path)
        figure_counts.append(count_figures(path))
        generate_chart(figure_counts, output_dir)
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            titulos.append(filename)
            if filename.endswith(".xml"):
                filepath = os.path.join(path, filename)
                figure_counts.append(count_figures(filepath))
        generate_chart(figure_counts, output_dir)
    else:
        print("Invalid path provided.")
        return
    

if __name__ == "__main__":
    # Check ih the user provided enough arguments
    if len(sys.argv) < 2:
        print("Use: python charts.py <folder_with_xmls> <output_folder>")
        sys.exit(1)

    path = sys.argv[1]    
    output_dir = sys.argv[2]
    process_files(path, output_dir)

You can use any folder to store the PDFs to be processed and any other to extract the results. They don’t have to be specifically named `paper`, `data`, or `results`; you just need to specify them when running the commands.
## Using GROBID for XML Extraction
To extract structured XML data from PDFs using [GROBID](https://github.com/kermitt2/grobid), follow these steps:

1. Start the [GROBID](https://github.com/kermitt2/grobid) container
Run the following command to launch a [GROBID](https://github.com/kermitt2/grobid) server using Docker:

```bash
docker run --rm -p 8070:8070 lfoppiano/grobid:latest-full
```
This will start the [GROBID](https://github.com/kermitt2/grobid) service on port 8070.

2. Process PDFs with [GROBID](https://github.com/kermitt2/grobid)
Once the [GROBID](https://github.com/kermitt2/grobid) server is running, you can extract XML from a folder of PDFs using the following command:

```bash
curl -F input=@<folder_with_pdf> "http://localhost:8070/api/processFulltextDocument" -o <output_xml>
```
Alternatively, for batch processing of all PDFs in a directory:

```bash
for file in <pdf_folder>/*.pdf; do
    curl -F input=@$file "http://localhost:8070/api/processFulltextDocument" -o "<output_folder>/$(basename "$file" .pdf).xml"
done
```
## Script Execution
### Extract entities and metadata from papers
Extracts information such as authors, organizations, keywords, and metadata from GROBID-generated XML files. We use the folder `xml_papers` as an input

**Command:**
```bash
python scripts/Text_Extraction.py <xml_folder>
```
**Output:** `output_files/Text_Extraction_results.json`

### Compute similarities and generate RDF  
Extracts abstracts, detects topics, computes similarity, and generates the semantic RDF.

**Command:**
```bash
python scripts/similarity_analysis.py 
```
**Output:** `output_files/similarity_matrix.png` `output_files/dendrogram.png` `output_files/similarity_data.rdf`


### Complete RDF graph  
Generates the final knowledge graph including metadata, entities, topics, and similarity relationships.

**Command:**
```bash
python scripts/dict_to_rdf.py 
```
**Output:** `output_files/knowledge_graph_linked.rdf` 

### Web visualization
Runs the interactive Streamlit demo to explore the knowledge graph.

**Command:**
```bash
streamlit run app.py
```
**Input:** `output_files/knowledge_graph_linked.rdf` `output_files/similarity_data.rdf`

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
curl -F input=@<path_to_pdf> "http://localhost:8070/api/processFulltextDocument" -o <output_xml>
```
Alternatively, for batch processing of all PDFs in a directory:

```bash
for file in <pdf_folder>/*.pdf; do
    curl -F input=@$file "http://localhost:8070/api/processFulltextDocument" -o "<output_folder>/$(basename "$file" .pdf).xml"
done
```

## Generate Keyword Cloud  
Extracts keywords from abstracts in XML files and creates a word cloud.

**Command:**
```bash
python scripts/keywordCloud.py <folder_with_xmls> <output_folder>
```
**Output:** `<output_folder>/keywordCloud.jpg`

## Chart Figures Count  
Counts the number of figures in each XML file and generates a bar chart.

**Command:**
```bash
python scripts/charts.py <folder_with_xmls> <output_folder>
```
**Output:** `<output_folder>/charts.jpg`

## Extract Links  
Extracts links from XML files while ignoring references.

**Command:**
```bash
python scripts/list.py <folder_with_xmls> <output_folder>
```
**Output:** `<output_folder>/links.txt`

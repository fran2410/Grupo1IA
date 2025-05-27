# AI-Open-Science

[![License](http://img.shields.io/:license-apache-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)
[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/) 
[![DOI](https://zenodo.org/badge/927066469.svg)](https://doi.org/10.5281/zenodo.14882666) 
[![Documentation Status](https://readthedocs.org/projects/ai-open-science/badge/?version=latest)](https://ai-open-science.readthedocs.io/en/latest/) 
[![GitHub release](https://img.shields.io/github/release/fran2410/AI-Open-Science.svg)](https://github.com/fran2410/AI-Open-Science/releases/)

## Description

This repository provides tools for extracting and visualizing information from scientific papers in XML format. Using [GROBID](https://github.com/kermitt2/grobid). for document processing, the scripts generate keyword clouds, charts displaying the number of figures per document, and extract links from XML files.

## Features
Given a XML file (or a directory with some of them) the tool will extract the data and make:
- **Keyword Cloud**: Keyword cloud based on the abstract information.
- **Charts**: Charts visualization showing the number of figures per article.
- **Links**: list of the links found in each paper while ignoring references.

## Project Structure

```
├── papers/              # Example research papers
├── data/                # Example XML files 
├── results/             # Example directory for generated files
├── scripts/             # Python scripts for data extraction and visualization
│   ├── keywordCloud.py  # Generates a keyword cloud from abstracts
│   ├── charts.py        # Creates charts showing the number of figures per document
│   ├── list.py          # Extracts links from XML files (excluding references)
├── docs/                # Additional documentation 
├── tests/               # Tests to check functionality 
```

# Installing fron Github

##  Clone the repository:
   ```bash
   git clone https://github.com/fran2410/AI-Open-Science.git
   cd AI-Open-Science
   ```
## 1. Conda

For installing Conda on your system, please visit the official Conda documentation [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

#### Create and activate the Conda environment
```bash
conda create -n ai-open-science python=3.13 
conda activate ai-open-science
```

## 2. Poetry

For installing Poetry on your system, please visit the official Poetry documentation [here](https://python-poetry.org/docs/#installation).

#### Install project dependencies
Run the following command in the root of the repository to install dependencies:
```bash
poetry install
```

# Installing through Docker

We provide a Docker image with the scripts already installed. To run through Docker, you may build the Dockerfile provided in the repository by running:

```bash
docker build -t ai-open-science .
```

Then, to run your image just type:

```bash
docker run --rm -it  ai-open-science
```

And you will be ready to use the scripts (see section below). If you want to have access to the results we recommend [mounting a volume](https://docs.docker.com/storage/volumes/). For example, the following command will mount the current directory as the `out` folder in the Docker image:

```bash
docker run -it --rm -v $PWD/out:/AI-Open-Science/out ai-open-science 
```
If you move any files produced by the scripts or set the output folder to `/out`, you will be able to see them in your current directory in the `/out` folder.

# USAGE
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

## Examples

For a sample execution with provided XML data, see the `results/` directory or run the scripts with sample files in `data/`.

## Where to Get Help

For any issues or questions, please open an issue in the [project issues](https://github.com/fran2410/AI-Open-Science/issues).

## Acknowledgements

Special thanks to the developers of [GROBID](https://github.com/kermitt2/grobid) for their tool for processing scientific documents.

## License

This project is distributed under the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0). Contributions to the project must follow the same licensing terms.


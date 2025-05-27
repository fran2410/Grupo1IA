
[![License](http://img.shields.io/:license-apache-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)
[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/) 
[![DOI](https://zenodo.org/badge/927066469.svg)](https://doi.org/10.5281/zenodo.14882666) 
[![Documentation Status](https://readthedocs.org/projects/ai-open-science/badge/?version=latest)](https://ai-open-science.readthedocs.io/en/latest/) 
[![GitHub release](https://img.shields.io/github/release/fran2410/AI-Open-Science.svg)](https://github.com/fran2410/AI-Open-Science/releases/)


## Description

This repository provides tools for extracting and visualizing information from scientific papers in XML format. Using [GROBID](https://github.com/kermitt2/grobid). for document processing, the scripts generate keyword clouds, charts displaying the number of figures per document, and extract links from XML files.

!!! info
    For any issues or questions, please open an issue in the [project issues](https://github.com/fran2410/AI-Open-Science/issues).

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

## Used Technologies and Standards

### Grobid
We use [GROBID](https://github.com/kermitt2/grobid) to process scientific documents and extract their XML files for use as data input.








# Rationale

## Project Motivation

The exponential growth of scientific literature has made manual exploration of research increasingly unmanageable. To address this challenge, this project proposes an automated pipeline that leverages Natural Language Processing (NLP) and Semantic Web technologies to extract, enrich, and visualize knowledge from scientific publications in PDF format.

## Objectives

- **Automate** the processing of scientific articles (PDF) into structured and semantically meaningful representations.
- **Extract** relevant metadata, named entities, topics, and document similarities.
- **Generate** an RDF knowledge graph that connects papers with authors, topics, and other related entities.
- **Visualize** the information interactively to facilitate exploration and discovery.

## Tool Selection Rationale

- **GROBID** was selected for its proven accuracy in extracting structured metadata and full text from scientific PDFs, especially generating well-formed TEI XML outputs.
- **HuggingFace Transformers** offers state-of-the-art NLP models (e.g., for Named Entity Recognition), enabling precise identification of authors, organizations, and other entities within acknowledgments.
- **Sentence Transformers** and **KeyBERT** were used to semantically analyze abstracts via embeddings, allowing the computation of document similarity and automatic topic detection.
- **rdflib** is a mature and flexible Python library for building and manipulating RDF graphs, making it ideal for the semantic representation layer of the project.
- **Streamlit** was used to create a lightweight, interactive demo for visualizing the resulting RDF knowledge graph and semantic insights.

## Pipeline Design

The overall workflow was designed as a modular pipeline with the following stages:

1. **PDF Preprocessing**: All papers are processed with GROBID to extract structured TEI XML.
2. **Text & Metadata Extraction**: Named entities and metadata are extracted using a dedicated script (`Text_Extraction.py`) and stored in JSON format.
3. **Topic Modeling & Similarity Analysis**: Abstracts are encoded and analyzed with `similarity_analysis.py` to derive topic distributions and inter-paper similarity metrics.
4. **RDF Graph Construction**: A comprehensive graph is built with `dict_to_rdf.py`, integrating metadata, authors, organizations, topics, and similarity links.
5. **Web-based Exploration**: Using `app.py` and Streamlit, users can interactively browse the RDF graph and explore semantic relationships.

## Open Science Practices

This project adheres to open science principles by:

- Releasing all code and documentation in a public GitHub repository.
- Relying exclusively on open-source tools and publicly available models.
- Providing detailed metadata and intermediate results for reproducibility.
- Including manual validation data to assess the performance of entity recognition models.

## Intended Impact

By combining NLP, embeddings, and semantic technologies, the project facilitates structured exploration of scientific corpora, supporting researchers in tasks such as:

- Discovering similar research works.
- Identifying key contributors and affiliations.
- Understanding topic distributions across a collection of papers.
- Building foundations for automatic recommendation or trend analysis systems.


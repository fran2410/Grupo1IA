FROM python:3.13

# Clonar el repositorio
RUN git clone https://github.com/fran2410/GRUPO1IA.git

# Instalar Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

RUN pip install poetry

WORKDIR "/GRUPO1IA"

# Instalar dependencias con Poetry
RUN poetry install

# Agregar instrucciones de uso y abrir bash
CMD ["/bin/bash", "-c", "echo 'Usage:\n  poetry run python scripts/keywordCloud.py <folder_with_xmls> <output_folder>\n  poetry run python scripts/charts.py <folder_with_xmls> <output_folder>\n  poetry run python scripts/list.py <folder_with_xmls> <output_folder>\n' && exec bash"]

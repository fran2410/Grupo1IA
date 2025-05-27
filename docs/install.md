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

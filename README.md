# Unrecognized Requests Clustering

This project analyzes unrecognized text requests by clustering them based on their semantic embeddings. It uses state-of-the-art NLP models and clustering strategies to automatically group similar requests and extract representative sentences for each cluster.

## Features

- **Embedding Generation:** Uses the [sentence-transformers](https://www.sbert.net/) model `all-MiniLM-L6-v2` to encode input texts.
- **Clustering:** Assigns requests to clusters based on cosine similarity between embeddings, iteratively reassign until changes fall below a specified threshold. Clusters with less than a specified minimum size are removed.
- **Cluster Naming:** Generates descriptive cluster names using n-gram frequency analysis with sklearn's `CountVectorizer`.
- **Representative Selection:** Applies Maximal Marginal Relevance to select diverse representative sentences from each cluster.
- **Evaluation:** Compares the clustering solution with an example solution using functions in `compare_clustering_solutions.py`.

## Project Structure
```
├── .idea/ # IDE configuration files (ignored)
├── data/ # Input CSV files with unrecognized requests
├── output/ # Generated clustering results (JSON format)
├── README.md
├── compare_clustering_solutions.py # Script for evaluating clustering results 
├── config.json # Configuration file 
└── main.py # Main entry point that reads config, clusters requests, and outputs results 
```

## Setup

1. **Python Environment:**  
   Ensure you have Python 3.8+ installed.

2. **Dependencies:**  
   Install the required Python packages. For example, you can create a virtual environment and install dependencies via pip:
   
   ```sh
   pip install pandas numpy scikit-learn torch sentence-transformers

3. **Configuration:**

Update the `config.json` file with the correct paths and parameters. The configuration includes:

- **data_file**: CSV file with the unrecognized requests (located in the `data/` directory).
- **output_file**: The file where the clustering results will be saved (located in the `output/` directory).
- **num_of_representatives**: Number of representative requests to be selected for each cluster.
- **min_cluster_size**: Minimum number of requests for a valid cluster.
- **example_solution_file**: File used for evaluating the clustering quality.

## Running the Project

Execute the main script from the command line:

```sh
python main.py
```
The script will:

- Read configuration from config.json.
- Process the unrecognized requests.
- Cluster the requests based on their semantic similarity.
- Save the clustering results to the specified output file.
- Evaluate the clustering solution by comparing it against an example solution.

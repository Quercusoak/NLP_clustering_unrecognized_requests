import json
from compare_clustering_solutions import evaluate_clustering
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sentence_transformers import SentenceTransformer


def extract_cluster_name(requests):
    words = " ".join(requests).split()
    common_words = [word for word, _ in Counter(words).most_common(3)]
    return " ".join(common_words)


def init_clusters(embeddings, min_size):
    """Use DBSCAN to initialize clusters, calculate their centroids, and update cluster labels"""

    # Get initial clusters from DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=min_size)
    labels = dbscan.fit_predict(embeddings)

    # Initialize incremental clusters
    clusters = {}
    cluster_assignments = labels.tolist()

    for i, label in enumerate(labels):
        if label != -1:
            clusters.setdefault(label, []).append(embeddings[i])

    centroids = {label: np.mean(clusters[label], axis=0) for label in clusters}

    return clusters, centroids, cluster_assignments


def add_embedding_to_cluster(model, cluster_assignments, embeddings, clusters, centroids):
    threshold = 0.65

    for i, embedding in enumerate(embeddings):
        # Get cosine similarity between embedded request and centroids
        similarities = model.similarity(embedding, list(centroids.values()))[0]
        max_index = np.argmax(similarities).item()  # index of the closest cluster
        
        # If embedded request is close to existing cluster - add and recalculate centroid
        if similarities[max_index] >= threshold:
            cluster_label = int(list(centroids.keys())[max_index])
            clusters[cluster_label].append(embedding)
            centroids[cluster_label] = np.mean(clusters[cluster_label], axis=0)
        else:
            cluster_label = int(max(clusters.keys(), default=-1) + 1)
            clusters[cluster_label] = [embedding]
            centroids[cluster_label] = embedding

        # Update embedding's cluster label
        cluster_assignments[i] = cluster_label
    

def remove_small_clusters(cluster_assignments, clusters, centroids, min_size):
    """remove clusters with less than min_size members"""
    to_remove = [label for label, cluster in clusters.items() if len(cluster) < min_size]

    for i, label in enumerate(cluster_assignments):
        if label in to_remove:
            cluster_assignments[i] = -1

    for i in to_remove:
        clusters.pop(i)
        centroids.pop(i)


def create_clusters(requests, min_size):
    """
    generate clusters and centroids:
    iterate over embeddings, where each request can be assigned to an existing cluster
    (if its proximity to the cluster’s centroid meets some similarity threshold you will find),
    otherwise the request initiates its own new cluster

    perform additional iterations over all request embeddings, re-assigning them if needed
    • till the algo convergence or till the max number of iterations is exhausted
    """

    # encode a set of unhandled requests using the sentence-transformers library
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(requests)

    # Use DBSCAN to initialize clusters, and calculate their centroids
    clusters, centroids, cluster_assignments = init_clusters(embeddings, min_size)

    # Incremental clustering refinement
    max_iterations = 10
    for iteration in range(max_iterations):
        old_assignments = cluster_assignments.copy()

        print(f'Iteration {iteration} start: {len(clusters)} clusters')

        # Iterate over embeddings and add them to clusters
        add_embedding_to_cluster(model, cluster_assignments, embeddings, clusters, centroids)

        # remove clusters with less than min_size members
        remove_small_clusters(cluster_assignments, clusters, centroids, min_size)

        print(f"Iteration {iteration} end: {len(clusters)} clusters")

        # early stopping: if clusters aren't changed or added
        max_changes = 5
        num_changes = sum(1 for a, b in zip(old_assignments, cluster_assignments) if a != b)
        print(f'num changes = {num_changes}')
        if num_changes <= max_changes:
            print(f"Early stop after {iteration} iterations")
            break

    return cluster_assignments


def analyze_unrecognized_requests(data_file, output_file, num_representatives, min_size):
    # todo: implement this function
    #  you are encouraged to split the functionality into multiple functions,
    #  but don't split your code into multiple *.py files
    #
    #  todo: the final outcome is the json file with clustering results saved as output_file

    # read data file into requests list
    csvfile = pd.read_csv(data_file)
    requests = csvfile['text'].values.tolist()

    min_size = int(min_size)
    num_representatives = int(num_representatives)

    cluster_assignments = create_clusters(requests, min_size)

    final_clusters = {}
    for i, cluster_id in enumerate(cluster_assignments):
        if cluster_id != -1:
            final_clusters.setdefault(cluster_id, []).append(requests[i])

    cluster_list = []
    for cluster_id, reqs in final_clusters.items():
        cluster_name = extract_cluster_name(reqs)
        representatives = reqs[:num_representatives]
        cluster_list.append({
            "cluster_name": cluster_name,
            "requests": reqs,
            "representatives": representatives
        })

    outliers_sentences = [requests[i] for i, label in enumerate(cluster_assignments) if label == -1]

    print(f'num clusters: {len(final_clusters)},num outliers: {len(outliers_sentences)}')

    # only clusters with size >= min_size are considered as generated clusters
    result = {"cluster_list": cluster_list, "unclustered": outliers_sentences}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    pass


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    # evaluate_clustering(config['example_solution_file'], config['example_solution_file'])  # invocation example
    #evaluate_clustering(config['example_solution_file'], config['output_file'])

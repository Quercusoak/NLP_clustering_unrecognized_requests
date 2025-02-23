import json

from compare_clustering_solutions import evaluate_clustering
import pandas as pd
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer


def extract_cluster_name(requests):
    words = " ".join(requests).split()
    common_words = [word for word, _ in Counter(words).most_common(3)]
    return " ".join(common_words)


def get_representatives(num_representatives, clusters: dict[int, list], centroids: dict, requests):
    """ Pick unique representing sentences from next to centroid and from edge of cluster. """

    representatives =[]

    # Select half of representatives from center of cluster
    num_center = int(num_representatives/2+1)

    # Select other half of representatives from the edges of the cluster
    num_edge = num_representatives - num_center

    cluster_labels = list(clusters.keys())

    for cluster_label, embedded_requests in clusters.items():
        centroid = centroids[cluster_label]
        similarities = get_similarities_list(centroid, embedded_requests)  # each embedded request's proximity to centroid
        # sort proximity to centroid desc
        sorted_indexes = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)

        # Step 1: Pick the most central point
        # central_idx = np.argmax(similarities)
        # representatives = [cluster_requests[central_idx]]
        # chosen_indices = {central_idx}


def add_embedding_to_cluster(embeddings_cluster_assignment, embeddings, clusters, centroids):
    """
    Using cosine similarity finds closest centroid to embedded request
    If proximity to centroid higher than threshold - assign to cluster and recalculate centroid
    otherwise the request initiates its own new cluster
    """
    threshold = 0.65  # Best value I found

    for i, embedding in enumerate(embeddings):
        # Get cosine similarity between embedded request and centroids
        # cluster_label, cosine_similarity = calc_similarity(embedding, centroids)

        similarities = get_similarities_list(embedding, list(centroids.values()))
        closest_centroid_idx = np.argmax(similarities).item()
        cluster_label = int(list(centroids.keys())[closest_centroid_idx])
        cosine_similarity = similarities[closest_centroid_idx]

        # Check if request close enough to existing cluster, or it starts a new cluster as the centroid
        if cosine_similarity >= threshold:
            clusters[cluster_label].append(embedding)
            centroids[cluster_label] = np.mean(clusters[cluster_label], axis=0)
        else:
            cluster_label = int(max(clusters.keys(), default=-1) + 1)
            clusters[cluster_label] = [embedding]
            centroids[cluster_label] = embedding

        # Update embedding's cluster label
        embeddings_cluster_assignment[i] = cluster_label
    

def remove_small_clusters(cluster_assignments, clusters, centroids, min_size):
    """remove clusters with less than min_size members"""
    to_remove = [label for label, cluster in clusters.items() if len(cluster) < min_size]

    for i, label in enumerate(cluster_assignments):
        if label in to_remove:
            cluster_assignments[i] = -1

    for i in to_remove:
        clusters.pop(i)
        centroids.pop(i)


def get_similarities_list(embedding, centroids: list):
    similarities = []
    for centroid in centroids:
        cosine = np.dot(embedding, centroid)
        similarities.append(cosine)

    return similarities


def similarities_list(embedding, centroids):
    similarities = []
    for cluster, centroid in centroids.items():
        cosine = np.dot(embedding, centroid)
        similarities.append(cosine)

    return similarities


def calc_similarity(embedding, centroids):
    """
    The embeddings are normalized, and so are centroids
    so for cosine similarity we calculate dot product and select closest centroid
    and the cluster the centroid belongs to
    """
    closest_cluster = (0, -1)

    for cluster, centroid in centroids.items():
        cosine = np.dot(embedding, centroid)
        if cosine > closest_cluster[1]:
            closest_cluster = (cluster, cosine)

    return closest_cluster


def create_clusters(requests, min_size):
    """
    iterate over embeddings, where each request can be assigned to an existing cluster
    (if its proximity to the cluster’s centroid meets some similarity threshold you will find),
    otherwise the request initiates its own new cluster

    perform additional iterations over all request embeddings, re-assigning them if needed
    till the algo convergence or till the max number of iterations is exhausted
    """

    # encode a set of unhandled requests using the sentence-transformers library
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(requests, normalize_embeddings=True)

    # Unassigned embeddings start with '-1' and then get cluster assignment
    embeddings_cluster_assignment = [-1 for _ in range(len(embeddings))]

    # Init first cluster (clustering algorithm can't start with empty centroids dict)
    clusters = {0: []}
    centroids = {0: embeddings[0]}

    # Incremental clustering refinement
    max_iterations = 10
    for iteration in range(max_iterations):
        old_assignments = embeddings_cluster_assignment.copy()

        print(f'Iteration {iteration} start: {len(clusters)} clusters')

        # Iterate over embeddings and add them to clusters
        add_embedding_to_cluster(embeddings_cluster_assignment, embeddings, clusters, centroids)

        # Remove clusters with less than min_size members
        remove_small_clusters(embeddings_cluster_assignment, clusters, centroids, min_size)

        print(f"Iteration {iteration} end: {len(clusters)} clusters")

        # Early stopping: if clusters aren't changed or added between iterations
        max_changes = 5
        num_changes = sum(1 for a, b in zip(old_assignments, embeddings_cluster_assignment) if a != b)
        print(f'num changes = {num_changes}')
        if num_changes <= max_changes:
            print(f"Early stop after {iteration} iterations")
            break

    return embeddings_cluster_assignment, clusters, centroids


def evaluate(output_file):
    with open(output_file, 'r') as sol_json_file:
        sol = json.load(sol_json_file)

    sol_clusters = sol['cluster_list']
    sol_unclustered = sol['unclustered']
    print(f'')


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

    cluster_assignments, clusters_embedded, centroids = create_clusters(requests, min_size)

    final_clusters = {}
    unclustered = []
    for i, cluster_id in enumerate(cluster_assignments):
        if cluster_id != -1:
            final_clusters.setdefault(cluster_id, []).append(requests[i])
        else:
            unclustered.append(requests[i])

    cluster_list = []
    for cluster_id, reqs in final_clusters.items():
        cluster_name = extract_cluster_name(reqs)
        representatives = reqs[:num_representatives]
        cluster_list.append({
            "cluster_name": cluster_name,
            "requests": reqs,
            "representatives": representatives
        })

    print(f'num clusters: {len(final_clusters)},num outliers: {len(unclustered)}')

    # only clusters with size >= min_size are considered as generated clusters
    result = {"cluster_list": cluster_list, "unclustered": unclustered}

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
    evaluate_clustering(config['example_solution_file'], config['output_file'])

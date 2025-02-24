import json

from compare_clustering_solutions import evaluate_clustering
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT


def extract_cluster_name(sentence_model, requests):
    kw_model = KeyBERT(model=sentence_model)
    doc = " ".join(requests)
    cluster_name = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=1)

    return cluster_name[0][0]


def get_cluster_representatives(num_representatives, embedded_requests, centroid):
    """ Pick unique representing sentences from next to centroid and from edge of cluster. """

    representatives = []

    # Get each request's proximity to centroid
    similarities = get_proximity_to_centroid(centroid, embedded_requests)
    requests_sorted = [req for _, req in sorted(similarities, key=lambda x: x[0], reverse=True)]

    num_requests = len(requests_sorted)

    # sort proximity to centroid desc
    # sorted_indexes = sorted(range(num_requests), key=lambda i: similarities_indexes[i][1], reverse=True)

    other_option = set()
    for r in requests_sorted:
        other_option.add(r)
        if len(other_option) >= num_representatives:
            break

    # Select half of representatives from center of cluster
    num_center = int(num_representatives / 2 + 1)
    a = 0
    while len(representatives) < num_center:
        res = requests_sorted[a]
        if res not in representatives:
            representatives.append(res)
        a+=1

    # Select other half of representatives from the edges of the cluster
    # num_edge = num_representatives - num_center
    a=0
    while len(representatives) < num_representatives:
        res = requests_sorted[num_requests - 1 - a]
        if res not in representatives:
            representatives.append(res)
        a+=1

    return representatives


def add_embedding_to_cluster(embeddings_cluster_assignment, embeddings, clusters, centroids):
    """
    Using cosine similarity finds closest centroid to embedded request
    If proximity to centroid higher than threshold - assign to cluster and recalculate centroid
    otherwise the request initiates its own new cluster
    """
    threshold = 0.65  # Best value I found

    for i, embedding in enumerate(embeddings):
        # Get cosine similarity between embedded request and centroids
        cluster_label, cosine_similarity = calc_similarity(embedding, centroids)

        # Check if request close enough to existing cluster, or it starts a new cluster as the centroid
        if cosine_similarity >= threshold:
            if embeddings_cluster_assignment[i] == cluster_label:
                continue
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


def get_proximity_to_centroid(centroid, embeddings: list):
    similarities = []
    for request, embedding in embeddings:
        cosine = np.dot(centroid, embedding)
        similarities.append((cosine, request))

    return similarities


def calc_similarity(embedding, centroids):
    """
    The embeddings and centroids are normalized, so for cosine similarity we calculate dot product
    returns closest cluster and similarity value
    """
    closest_cluster = (0, -1)

    for cluster, centroid in centroids.items():
        cosine = np.dot(embedding, centroid)
        if cosine > closest_cluster[1]:
            closest_cluster = (cluster, cosine)

    return closest_cluster


def create_clusters(model, requests, min_size):
    """
    iterate over embeddings, where each request can be assigned to an existing cluster
    (if its proximity to the cluster’s centroid meets some similarity threshold you will find),
    otherwise the request initiates its own new cluster

    perform additional iterations over all request embeddings, re-assigning them if needed
    till the algo convergence or till the max number of iterations is exhausted
    """
    # encode a set of unhandled requests using the sentence-transformers library
    embeddings = model.encode(requests, normalize_embeddings=True)

    # Unassigned embeddings start with '-1' and then get cluster assignment
    embeddings_cluster_assignment = [-1 for _ in range(len(embeddings))]

    requests_to_embeddings = [(r, e) for r, e in zip(requests, embeddings)]

    # Init first cluster (clustering algorithm can't start with empty centroids dict)
    clusters = {0: [requests_to_embeddings[0]]}
    centroids = {0: embeddings[0]}
    embeddings_cluster_assignment[0] = 0

    # Incremental clustering refinement
    max_iterations = 10
    for iteration in range(max_iterations):
        old_assignments = embeddings_cluster_assignment.copy()

        print(f'Iteration {iteration} start: {len(clusters)} clusters')

        # Iterate over embeddings and add them to clusters
        # add_embedding_to_cluster(embeddings_cluster_assignment, embeddings, clusters, centroids)
        threshold = 0.65  # Best value I found

        for i, request_embedding in enumerate(requests_to_embeddings):
            request, embedding = request_embedding
            prev_cluster = embeddings_cluster_assignment[i]

            # Get cosine similarity between embedded request and centroids
            cluster_label, cosine_similarity = calc_similarity(embedding, centroids)

            # If assignment unchanged - continue
            if cosine_similarity >= threshold and prev_cluster == cluster_label:
                continue

            # Remove previous assignment if there was one
            if prev_cluster != -1:
                clusters[prev_cluster].remove(request_embedding)

            # If request is not close enough to existing cluster, it starts a new cluster
            if cosine_similarity < threshold:
                cluster_label = int(max(clusters.keys(), default=-1) + 1)

            # Update request's cluster assignment, add to cluster and recalculate centroid
            embeddings_cluster_assignment[i] = cluster_label
            clusters.setdefault(cluster_label, []).append(request_embedding)
            cluster_embeddings = [e for _, e in clusters[cluster_label]]
            centroids[cluster_label] = np.mean(cluster_embeddings, axis=0)

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

    # encode a set of unhandled requests using the sentence-transformers library
    model = SentenceTransformer("all-MiniLM-L6-v2")

    min_size = int(min_size)
    num_representatives = int(num_representatives)

    cluster_assignments, clusters, centroids = create_clusters(model, requests, min_size)

    unclustered = [requests[i] for i, cluster in enumerate(cluster_assignments) if cluster == -1]

    cluster_list = []
    for cluster, reqs_embedding in clusters.items():
        cluster_name = extract_cluster_name(model, [req for req, _ in reqs_embedding])
        representatives = get_cluster_representatives(num_representatives, reqs_embedding, centroids[cluster])
        cluster_list.append({
            "cluster_name": cluster_name,
            "requests": [req for req, _ in reqs_embedding],
            "representatives": representatives
        })

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

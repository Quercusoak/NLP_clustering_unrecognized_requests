import json
from operator import itemgetter

from compare_clustering_solutions import evaluate_clustering
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, SimilarityFunction
from keybert import KeyBERT


def extract_cluster_name(kw_model, requests, embeddings):
    # doc = " ".join(requests)
    cluster_name = kw_model.extract_keywords(
        requests, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=1, doc_embeddings=embeddings)

    return cluster_name[0][0]


def get_cluster_representatives(model, num_representatives, embeddings, requests, centroid):
    """
    Pick unique and diverse representing sentences for cluster using Maximal Marginal Relevance.
    For each request calculate mmr - proximity to centroid vs. distance from other selected requests.
    """
    representatives = []
    diversity = 0.7
    embeddings = np.array(embeddings)

    relevance_scores = model.similarity(centroid, embeddings)[0]
    embedding_similarities = model.similarity(embeddings, embeddings)

    # Select request closest to the centroid
    selected_idx = [np.argmax(relevance_scores).item()]
    representatives.append(requests[selected_idx[0]])

    # remaining_idx = list(set(range(len(requests))) - set(selected_idx))
    remaining_idx = [i for i in range(len(embeddings)) if i != selected_idx[0]]

    # Diversity score is min similarity to selected requests

    for _ in range(num_representatives - 1):
        # candidate_similarities = relevance_scores[remaining_idx]
        # a = embedding_similarities[remaining_idx][:, selected_idx]
        # target_similarities = max(embedding_similarities[remaining_idx][:, selected_idx])

        # Compute diversity score: min similarity to already selected requests
        diversity_scores = np.array([
            min(model.similarity(embeddings[i], embeddings[selected_idx])[0])
            for i in remaining_idx
        ])

        # Select request by MMR score: balance between relevance and diversity
        mmr_scores = (1 - diversity) * relevance_scores[remaining_idx] - diversity * diversity_scores
        # mmr_scores = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = remaining_idx[np.argmax(mmr_scores)]

        selected_idx.append(mmr_idx)
        representatives.append(requests[mmr_idx])
        remaining_idx.remove(mmr_idx)

    return representatives


def add_embedding_to_cluster(embeddings_cluster_assignment, requests_to_embeddings, clusters, centroids):
    """
    Using cosine similarity finds closest centroid to embedded request
    If proximity to centroid higher than threshold - assign to cluster and recalculate centroid
    otherwise the request initiates its own new cluster
    """
    threshold = 0.65  # Best value I found

    for i, request_embedding in enumerate(requests_to_embeddings):
        request, embedding = request_embedding
        prev_cluster = embeddings_cluster_assignment[i]

        # Get closest cluster by cosine similarity between embedded request and centroids
        cluster_label, cosine_score = calc_similarity(embedding, centroids)

        # If assignment unchanged - continue
        if cosine_score >= threshold and prev_cluster == cluster_label:
            continue

        # Remove previous assignment if there was one
        if prev_cluster != -1:
            clusters[prev_cluster].remove(request_embedding)

        # If request is not close enough to existing cluster, it starts a new cluster
        if cosine_score < threshold:
            cluster_label = int(max(clusters.keys(), default=-1) + 1)

        # Update request's cluster assignment, add to cluster and recalculate centroid
        embeddings_cluster_assignment[i] = cluster_label
        clusters.setdefault(cluster_label, []).append(request_embedding)
        cluster_embeddings = [e for _, e in clusters[cluster_label]]
        centroids[cluster_label] = np.mean(cluster_embeddings, axis=0)
    

def remove_small_clusters(cluster_assignments, clusters, centroids, min_size):
    """remove clusters with less than min_size members"""
    to_remove = [label for label, cluster in clusters.items() if len(cluster) < min_size]

    for i, label in enumerate(cluster_assignments):
        if label in to_remove:
            cluster_assignments[i] = -1

    for i in to_remove:
        clusters.pop(i)
        centroids.pop(i)


def get_proximity_to_centroid(centroid, embeddings):
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
    min_changes = 5
    for iteration in range(max_iterations):
        old_assignments = embeddings_cluster_assignment.copy()

        print(f'Iteration {iteration} start: {len(clusters)} clusters')

        # Iterate over embeddings and add them to clusters
        add_embedding_to_cluster(embeddings_cluster_assignment, requests_to_embeddings, clusters, centroids)

        # Remove clusters with less than min_size members
        remove_small_clusters(embeddings_cluster_assignment, clusters, centroids, min_size)

        print(f"Iteration {iteration} end: {len(clusters)} clusters")

        # Early stopping: if clusters aren't changed or added between iterations
        num_changes = sum(1 for a, b in zip(old_assignments, embeddings_cluster_assignment) if a != b)
        print(f'num changes = {num_changes}')
        if num_changes <= min_changes:
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
    model = SentenceTransformer("all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.DOT_PRODUCT)
    kw_model = KeyBERT(model=model)

    min_size = int(min_size)
    num_representatives = int(num_representatives)

    cluster_assignments, clusters, centroids = create_clusters(model, requests, min_size)

    unclustered = [requests[i] for i, cluster in enumerate(cluster_assignments) if cluster == -1]

    cluster_list = []
    for cluster, reqs_embedding in clusters.items():
        r = [req for req, _ in reqs_embedding]
        e = [em for _, em in reqs_embedding]
        cluster_name = extract_cluster_name(kw_model, r, e)
        representatives = get_cluster_representatives(model,num_representatives, e,r, centroids[cluster])
        cluster_list.append({
            "cluster_name": cluster_name,
            "requests": r,
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

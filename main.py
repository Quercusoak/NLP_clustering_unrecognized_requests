import json
from collections import Counter

from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer
from compare_clustering_solutions import evaluate_clustering
import numpy as np
from sentence_transformers import SentenceTransformer, SimilarityFunction
import torch


def cluster_name_ngrams(sentences):
    # CountVectorizer for cluster names with n-gram range
    vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words="english")
    try:
        X = vectorizer.fit_transform(sentences)
    except:
        vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words=None)
        X = vectorizer.fit_transform(sentences)

    # Get feature names (n-grams) and their counts, sort by first trigrams then bigrams
    ngram_counts = Counter(dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0))))
    sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: (-len(x[0].split()), -x[1]))

    best = sorted_ngrams[0]  # Select most frequent trigram - prefer trigram to bigram

    # Get the most frequent n-grams (most likely a bigram) and check if it is significantly more frequent
    most = ngram_counts.most_common(1)[0]
    if most[1] > 3 * (best[1]):
        best = most

    return best[0]


def get_cluster_representatives(model, num_representatives, embeddings, requests, centroid, diversity=0.5):
    """
    Pick unique and diverse representing sentences for cluster using Maximal Marginal Relevance.
    For each request calculate mmr - proximity to centroid vs. distance from other selected requests.
    """
    representatives = []
    embeddings = np.array(embeddings)

    # Cosine similarity to centroid shows relevance to cluster
    relevance_scores = model.similarity(centroid, embeddings)[0]

    # Cosine similarity of every request to all others in the cluster
    embedding_similarities = model.similarity(embeddings, embeddings)

    # Select request closest to the centroid
    selected_idx = [np.argmax(relevance_scores).item()]
    representatives.append(requests[selected_idx[0]])

    remaining_idx = [i for i in range(len(embeddings)) if i != selected_idx[0]]

    for _ in range(num_representatives - 1):
        candidate_similarities = relevance_scores[remaining_idx]
        # model.similarity returns tensors, so they need torch.min
        target_similarities = torch.min(embedding_similarities[remaining_idx][:, selected_idx], dim=1).values

        # Select request by MMR score: balance between relevance and diversity
        mmr_scores = (1 - diversity) * candidate_similarities - diversity * target_similarities
        mmr_idx = remaining_idx[np.argmax(mmr_scores)]

        selected_idx.append(mmr_idx)
        representatives.append(requests[mmr_idx])
        remaining_idx.remove(mmr_idx)

    return representatives


def add_embedding_to_cluster(embeddings_cluster_assignment, embeddings, clusters, centroids):
    """
    Using cosine similarity finds closest centroid to embedded request
    If proximity to centroid higher than threshold - assign to cluster and recalculate centroid
    otherwise the request initiates its own new cluster
    """
    threshold = 0.65  # Best value I found

    for i, embedding in enumerate(embeddings):
        prev_cluster = embeddings_cluster_assignment[i]

        # Get closest cluster by cosine similarity between embedded request and centroids
        cluster_label, cosine_score = calc_similarity(embedding, centroids)

        # If assignment unchanged - continue
        if cosine_score >= threshold and prev_cluster == cluster_label:
            continue

        # Remove previous assignment if there was one
        if prev_cluster != -1:
            clusters[prev_cluster] = remove_embedding(clusters[prev_cluster], embedding)

        # If request is not close enough to existing cluster, it starts a new cluster
        if cosine_score < threshold:
            cluster_label = int(max(clusters.keys(), default=-1) + 1)

        # Update request's cluster assignment, add to cluster and recalculate centroid
        embeddings_cluster_assignment[i] = cluster_label
        clusters.setdefault(cluster_label, []).append(embedding)
        centroids[cluster_label] = np.mean(clusters[cluster_label], axis=0)


def remove_embedding(cluster, embedding):
    """
    When two numpy arrays are compared a numpy array of boolean values (the element by element comparisons)
    is returned which is interpreted as being ambiguous.
    So a custom removal is needed.
    """
    for i, arr in enumerate(cluster):
        if np.array_equal(arr, embedding):  # Check element-wise equality
            del cluster[i]  # Remove first occurrence
            return cluster

def remove_small_clusters(cluster_assignments, clusters, centroids, min_size):
    """remove clusters with less than min_size members"""
    to_remove = [label for label, cluster in clusters.items() if len(cluster) < min_size]

    for i, label in enumerate(cluster_assignments):
        if label in to_remove:
            cluster_assignments[i] = -1

    for i in to_remove:
        clusters.pop(i)
        centroids.pop(i)


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


def init_clusters(embeddings):
    # Unassigned embeddings start with '-1' and then get cluster assignment
    embeddings_cluster_assignment = [-1 for _ in range(len(embeddings))]

    # Init first cluster (clustering algorithm can't start with empty centroids dict)
    clusters = {0: [embeddings[0]]}
    centroids = {0: embeddings[0]}
    embeddings_cluster_assignment[0] = 0

    return clusters, centroids, embeddings_cluster_assignment


def create_clusters(embeddings, min_size):
    """
    In each iteration:
    Assigns requests to clusters if proximity to cendroid >= threshold, otherwise they start a new cluster.
    Then removed clusters < min_size.
    If number of request that changed assignment lower than min_changes - early stop.
    """
    clusters, centroids, embeddings_cluster_assignment = init_clusters(embeddings)

    max_iterations = 10
    min_changes = 10
    for iteration in range(max_iterations):
        old_assignments = embeddings_cluster_assignment.copy()

        # Iterate over embeddings and add them to clusters
        add_embedding_to_cluster(embeddings_cluster_assignment, embeddings, clusters, centroids)

        # Remove clusters with less than min_size members
        remove_small_clusters(embeddings_cluster_assignment, clusters, centroids, min_size)

        # Early stopping: if clusters aren't changed or added between iterations
        num_changes = sum(1 for a, b in zip(old_assignments, embeddings_cluster_assignment) if a != b)
        if num_changes <= min_changes:
            print(f"Early stop after {iteration} iterations")
            break

    return clusters, centroids, embeddings_cluster_assignment


def analyze_unrecognized_requests(data_file, output_file, num_representatives, min_size):
    # read data file into requests list
    csvfile = read_csv(data_file)
    requests = csvfile['text'].values.tolist()

    # Encode a set of unhandled requests using the sentence-transformers library
    model = SentenceTransformer("all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.DOT_PRODUCT)
    embeddings = model.encode(requests, normalize_embeddings=True)

    clusters, centroids, cluster_assignments = create_clusters(embeddings, int(min_size))

    final_clusters = {}
    unclustered = []
    for i, cluster_id in enumerate(cluster_assignments):
        if cluster_id != -1:
            final_clusters.setdefault(cluster_id, []).append(requests[i])
        else:
            unclustered.append(requests[i])

    cluster_list = []
    for l, reqs in final_clusters.items():
        representatives = get_cluster_representatives(model, int(num_representatives), clusters[l], reqs, centroids[l])
        cluster_name = cluster_name_ngrams(reqs)
        cluster_list.append({
            "cluster_name": cluster_name,
            "requests": reqs,
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

    evaluate_clustering(config['example_solution_file'], config['output_file'])

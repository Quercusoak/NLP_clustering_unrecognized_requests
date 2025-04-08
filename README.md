# NLP Clustering of Unrecognized Requests

This project applies unsupervised learning methods to cluster unrecognized requests from users. By grouping similar requests together, the system identifies common themes and potential areas for improvement in intent recognition. The approach leverages modern NLP techniques to generate high-quality clusters with descriptive titles and representative examples.

## Overview

The workflow of the project is as follows:

- Embed requests using SentenceTransformer.
- Group requests into clusters based on cosine similarity.
- Improves efficiency by halting clustering iterations when minimal changes are detected.
- Assign fitting titles to clusters using n-grams frequency analysis.
- Select diverse representative sentences for each cluster using Maximal Marginal Relevance (MMR).
- Evaluate the clustering solution.

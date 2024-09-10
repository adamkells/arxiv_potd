import arxiv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Query recent papers from a specific category (e.g., "cs.LG" for machine learning)
def fetch_recent_papers(category="cs.LG", max_results=100):
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    return [result for result in search.results()]

# Create embeddings for paper abstracts
def embed_abstracts(papers, model):
    abstracts = [paper.summary for paper in papers]
    embeddings = model.encode(abstracts)
    return embeddings

# Define a function to find the top N most relevant papers
def find_top_papers(query, embeddings, papers, model, top_n=10):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_papers = [papers[i] for i in top_indices]
    return top_papers

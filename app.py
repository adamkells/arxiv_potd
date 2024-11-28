from sentence_transformers import SentenceTransformer
import arxiv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import srsly
from dotenv import load_dotenv
import os
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Email configuration
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
RECIPIENT_LIST = os.getenv('RECIPIENT_LIST', '').split(',')

# File paths
DATA_DIR = Path('data')
POTD_FILE = DATA_DIR / 'potd.jsonl'

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)
if not POTD_FILE.exists():
    POTD_FILE.write_text('[]')

def fetch_recent_papers(max_results: int = 100) -> List[Dict[str, Any]]:
    """Fetch recent papers from arXiv."""
    search = arxiv.Search(
        query="cat:stat.ML OR cat:cs.LG",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = []
    for result in search.results():
        paper = {
            'title': result.title,
            'abstract': result.summary,
            'authors': [author.name for author in result.authors],
            'url': result.pdf_url,
            'published': result.published.strftime('%Y-%m-%d')
        }
        papers.append(paper)
    
    return papers

def embed_abstracts(papers: List[Dict[str, Any]], model: SentenceTransformer) -> np.ndarray:
    """Embed paper abstracts using the provided model."""
    abstracts = [paper['abstract'] for paper in papers]
    return model.encode(abstracts)

def find_top_papers(query: str, embeddings: np.ndarray, papers: List[Dict[str, Any]], 
                    model: SentenceTransformer, top_k: int = 5) -> List[Dict[str, Any]]:
    """Find top papers matching the query."""
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [papers[i] for i in top_indices]

def send_email(recipients: List[str], paper: Dict[str, Any]) -> None:
    """Send email with paper information."""
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = ', '.join(recipients)
    msg['Subject'] = f"Daily arXiv Paper: {paper['title']}"

    body = f"""
    Title: {paper['title']}
    Authors: {', '.join(paper['authors'])}
    Published: {paper['published']}

    Abstract:
    {paper['abstract']}

    URL: {paper['url']}
    """
    
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            print("Email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")

def main():
    try:
        # Load model
        print("Loading model...")
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Fetch papers
        print("Fetching recent papers...")
        recent_papers = fetch_recent_papers()
        
        # Embed abstracts
        print("Embedding abstracts...")
        embeddings = embed_abstracts(recent_papers, model)
        
        # Find top papers
        print("Finding top papers...")
        query = "causality for time series forecasting"
        top_papers = find_top_papers(query, embeddings, recent_papers, model, top_k=1)
        
        if top_papers:
            # Send email
            print("Sending email...")
            send_email(RECIPIENT_LIST, top_papers[0])
            
            # Save to history
            try:
                old_papers = list(srsly.read_jsonl(POTD_FILE))
                old_papers.append(top_papers[0])
                srsly.write_jsonl(POTD_FILE, old_papers)
                print("Paper saved to history")
            except Exception as e:
                print(f"Error saving to history: {e}")
                
    except Exception as e:
        print(f"Error in main process: {e}")

if __name__ == "__main__":
    main()

from flask import Flask, request, render_template
from paper_scraper import find_top_papers, fetch_recent_papers, embed_abstracts
from sentence_transformers import SentenceTransformer
import srsly

app = Flask(__name__)

recent_papers = fetch_recent_papers()
# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embed_abstracts(recent_papers, model)
top_papers = find_top_papers("causality for time series forecasting", embeddings, recent_papers, model)

@app.route('/')
def home():
    return render_template('index.html', papers=top_papers[:1])

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    top_papers = find_top_papers(query, embeddings, recent_papers, model)
    return render_template('results.html', papers=top_papers)

old_papers = list(srsly.read_jsonl('data/potd.jsonl'))
old_papers.append(top_papers[0])
# TODO: Reformat to make papers json seriablizable.
# srsly.write_jsonl('data/potd.jsonl', old_papers)

@app.route('/old_potd', methods=['POST'])
def old_potd():
    return render_template('results.html', papers=old_papers)


if __name__ == '__main__':
    app.run(debug=True)


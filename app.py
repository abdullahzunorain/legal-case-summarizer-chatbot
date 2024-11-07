import streamlit as st
import torch
import faiss
import numpy as np
from datasets import load_dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load a subset of the LEDGAR dataset for demonstration
dataset = load_dataset("lex_glue", "ledgar", split="train[:500]")
documents = [entry["text"] for entry in dataset]

# Load the retrieval model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retriever_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v3').to(device)
doc_embeddings = retriever_model.encode(documents, convert_to_tensor=True).cpu().numpy()

# Initialize FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Document retrieval function
def retrieve_documents(query, top_k=5):
    query_embedding = retriever_model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, top_k)
    results = [{"text": documents[idx], "score": distances[0][i]} for i, idx in enumerate(indices[0])]
    return results

# Load summarization model, set to GPU if available
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# Summarization function with dynamic length
def summarize_text(text):
    input_length = len(text.split())
    max_length = min(150, int(input_length * 0.6))
    min_length = min(30, int(input_length * 0.3))
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Retrieve and summarize function
def retrieve_and_summarize(query, top_k=5):
    retrieved_docs = retrieve_documents(query, top_k)
    summaries = []
    for doc in retrieved_docs:
        summary = summarize_text(doc['text'])
        summaries.append({"original": doc['text'], "summary": summary, "score": doc['score']})
    return summaries

# Streamlit app structure
st.title("Legal Case Summarizer")
query = st.text_input("Enter a legal topic or question:")
if query:
    st.write("Retrieving and summarizing...")
    results = retrieve_and_summarize(query, top_k=3)
    for i, result in enumerate(results):
        st.subheader(f"Document {i+1}")
        st.write("Original Text:", result['original'])
        st.write("Summary:", result['summary'])
        st.write("Score:", result['score'])

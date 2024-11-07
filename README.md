# Legal Case Summarizer

## Overview
The **Legal Case Summarizer** is a web application built with **Streamlit**, designed to help legal professionals and researchers quickly retrieve relevant legal documents and generate concise summaries. The application uses a combination of **document retrieval** and **summarization models** to process user queries, providing summarized text from a set of legal documents.

This project demonstrates the application of **Sentence Transformers**, **FAISS (Facebook AI Similarity Search)**, and **transformer-based models** for effective document retrieval and summarization.

## Features
- **Document Retrieval**: Users can input a legal topic or question, and the system will retrieve the most relevant legal documents from a pre-defined dataset.
- **Summarization**: Retrieved documents are then summarized using a transformer-based model (Facebook BART), which condenses the text to highlight key information.
- **Interactive Interface**: The app provides a user-friendly interface built with **Streamlit**, enabling seamless interaction and easy visualization of results.

## Technologies Used
- **Streamlit**: To build the web-based interface.
- **Transformers**: To leverage state-of-the-art NLP models like BART for text summarization.
- **Sentence-Transformers**: For generating embeddings of documents and queries to enable efficient retrieval.
- **FAISS**: For fast similarity search and efficient retrieval of relevant documents based on query embeddings.
- **Torch**: For deep learning computations and to support models that require GPU acceleration.

## How It Works
1. **Dataset**: The project utilizes a subset of the **LEDGAR dataset** from the **lex_glue** collection, which consists of legal texts. The dataset is used to build the document pool for retrieval.
2. **Document Retrieval**: When a user enters a query, the application generates an embedding of the query using a pre-trained **SentenceTransformer** model. The query embedding is then compared to the embeddings of the documents stored in a FAISS index. The most relevant documents are retrieved based on their similarity to the query.
3. **Summarization**: The retrieved documents are passed through a pre-trained **BART** summarization model, which condenses the content to create a brief, readable summary.
4. **User Interface**: The results (both original text and summary) are displayed in the Streamlit app, providing an interactive experience for users to explore and understand the legal information related to their query.

## Installation

To run this project locally, you'll need to install the required libraries. You can install them using the following command:

```bash
pip install -r requirements.txt
```

Make sure that you have Python 3.6+ and the necessary libraries installed:
- **Transformers**
- **Sentence-Transformers**
- **FAISS**
- **Torch**
- **Streamlit**
- **Datasets**

### Requirements File Example

```txt
streamlit
transformers
sentence-transformers
faiss-cpu
torch
datasets
```

## Running the App
To run the application, simply run the following command in your terminal:

```bash
streamlit run app.py
```

This will launch the app in your default web browser where you can start interacting with the tool.

## Model Details
- **Retrieval Model**: `msmarco-distilbert-base-v3` from **Sentence-Transformers** for document embedding and retrieval.
- **Summarization Model**: `facebook/bart-large-cnn` from **Transformers** for generating summaries of the retrieved legal texts.

## Limitations and Future Enhancements
- **Legal Domain Specificity**: The current model uses general transformer-based models. For better results, domain-specific models could be integrated for legal texts.
- **Performance**: The current setup runs on CPU, which may lead to slower performance for large datasets. GPU support can be integrated for faster computations.
- **Real-time Database Integration**: Instead of using a static subset of the **LEDGAR dataset**, real-time legal document databases could be integrated for more dynamic results.

## Contributing
Contributions are welcome! Feel free to fork the repository, submit issues, or create pull requests to improve the functionality, enhance the summarization quality, or add new features.

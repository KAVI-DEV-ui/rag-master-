# RAG Application

A Retrieval-Augmented Generation (RAG) application for document question answering using LangChain, FAISS, and Streamlit.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Ajaikumar0712/RAG.git
cd RAG
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv rag_env
rag_env\Scripts\activate

# macOS/Linux
python -m venv rag_env
source rag_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App
```bash
streamlit run src/app.py
```

### Running Individual Components
```bash
# Data preparation
python src/data_preparation.py

# Embeddings generation
python src/embeddings_store.py

# RAG pipeline
python src/rag_pipeline.py
```

## Project Structure
```
RAG/
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── data_preparation.py    # Document processing
│   ├── embeddings_store.py    # Vector embeddings management
│   └── rag_pipeline.py        # RAG implementation
├── data/
│   └── *.pdf                  # Document files
├── outputs/
│   ├── chunks.json           # Processed text chunks
│   └── faiss_index/          # Vector index files
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Features

- PDF document processing
- Vector embeddings with FAISS
- Interactive web interface with Streamlit
- Question answering with context retrieval

## Requirements

- Python 3.8+
-  gemini (models)

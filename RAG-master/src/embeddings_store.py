import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

CHUNKS_PATH = Path("../outputs/chunks.json")  
INDEX_PATH = Path("../outputs/faiss_index")  
def build_faiss_index(chunks_path=CHUNKS_PATH, index_path=INDEX_PATH):
    print(f"[INFO] Loading chunks from {chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    metadatas = [item["metadata"] for item in data]

    print(f"[INFO] Loaded {len(texts)} chunks for embedding.")

    
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

   
    vectorstore = FAISS.from_texts(texts, embedder, metadatas=metadatas)

    
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))

    print(f"[INFO] FAISS index saved to {index_path}")

if __name__ == "__main__":
    build_faiss_index()

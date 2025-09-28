import json
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


DATA_PATH = Path("../data/Microbial_Degradation_of_Microplastics_Mechanisms_.pdf")  
OUTPUT_PATH = Path("../outputs/chunks.json")    

def prepare_data(pdf_path=DATA_PATH, output_path=OUTPUT_PATH):
    print(f"[INFO] Loading PDF: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} pages.")

    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   
        chunk_overlap=200  
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Split into {len(chunks)} chunks.")

    
    data = [{"text": c.page_content, "metadata": c.metadata} for c in chunks]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved chunks to {output_path}")

if __name__ == "__main__":
    prepare_data()

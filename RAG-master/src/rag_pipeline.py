from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
import os


INDEX_PATH = Path("../outputs/faiss_index")

def run_rag(query: str):
    print("[INFO] Loading FAISS index...")
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(str(INDEX_PATH), embedder, allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    
    print("[INFO] Loading Gemini model...")
    
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
   
    
    if not api_key:
        print("Error: Please set the GOOGLE_API_KEY environment variable or add your API key directly in the script")
        print("You can get an API key from: https://aistudio.google.com/app/apikey")
        print("Then either:")
        print("1. Set environment variable: $env:GOOGLE_API_KEY='your-api-key-here'")
        print("2. Or uncomment and edit the api_key line in this script")
        return
    
   
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",  
        google_api_key="AIzaSyBFzlDYOrvhgOoCOr_h7uEfW2N9PArcR6g",
        temperature=0.1
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print(f"[USER QUERY]: {query}")
    answer = qa.invoke(query)
    print(f"[ANSWER]: {answer['result']}")  
if __name__ == "__main__":
    
    run_rag("What are the main findings in this paper?")

import streamlit as st
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI

# --- File paths ---
INDEX_PATH = Path("../outputs/faiss_index")

@st.cache_resource
def load_pipeline():
    st.write("[INFO] Loading FAISS index...")
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(str(INDEX_PATH), embedder, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # --- Use Gemini API ---
    st.write("[INFO] Loading Gemini model...")
    try:
        # Get API key from environment variable or set it directly for testing
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("Please set your GOOGLE_API_KEY environment variable")
            st.info("You can get an API key from: https://ai.google.dev/")
            return None
            
        llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1
        )
        
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        return qa
    except Exception as e:
        st.error(f"Error loading Gemini model: {str(e)}")
        return None

def main():
    st.title("🤖 RAG PDF Chatbot with Gemini")
    st.write("Ask questions about your PDF document and get AI-powered answers!")
    
    # Add sidebar for API key and information
    with st.sidebar:
        st.header("� API Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google API Key", 
            value=os.environ.get("GOOGLE_API_KEY", ""),
            type="password",
            help="Get your API key from https://ai.google.dev/"
        )
        
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.success("✅ API Key set!")
        else:
            st.warning("⚠️ Please enter your API key above")
            
        st.markdown("---")
        st.header("📋 System Information")
        st.write("**Model:** Google Gemini 1.5 Flash")
        st.write("**Vector DB:** FAISS")
        st.write("**Embeddings:** all-MiniLM-L6-v2")
        
        if st.button("🔄 Reload Model"):
            st.cache_resource.clear()
            st.rerun()

    # Only proceed if API key is provided
    if not api_key:
        st.error("❌ Please enter your Google API key in the sidebar to continue.")
        st.info("💡 You can get a free API key from: https://ai.google.dev/")
        return

    qa = load_pipeline()

    if qa is not None:
        st.success("✅ Model loaded successfully! You can now ask questions.")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask a question about your document..."):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = qa.invoke(prompt)
                        response = result["result"]
                        st.markdown(response)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error generating answer: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        st.error("❌ Failed to load the model. Please check the errors above.")

if __name__ == "__main__":
    main()

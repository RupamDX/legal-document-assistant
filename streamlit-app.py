import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download spaCy model at startup
import subprocess
import sys

def download_spacy_model():
    try:
        import spacy
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "spacy", 
                "download", 
                "en_core_web_sm"
            ])
    except Exception as e:
        st.error(f"Error loading spaCy model: {str(e)}")
        return None
    return spacy.load('en_core_web_sm')

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'extracted_info' not in st.session_state:
    st.session_state.extracted_info = None
if 'nlp' not in st.session_state:
    st.session_state.nlp = None

# Load Models
@st.cache_resource
def load_models():
    nlp = download_spacy_model()
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return nlp, embedding_model

# Initialize models
with st.spinner('Loading models...'):
    nlp, embedding_model = load_models()
    if nlp is None:
        st.error("Failed to load spaCy model. Please check the logs for details.")
        st.stop()
    st.session_state.nlp = nlp

# Function to Parse and Extract Key Information from PDFs
def parse_and_extract(uploaded_file):
    """Parse and extract key information from a PDF."""
    try:
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load the PDF document
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # Extract text from the document
        text = " ".join([doc.page_content for doc in documents])

        # Use spaCy for Named Entity Recognition (NER)
        doc = st.session_state.nlp(text)
        extracted_info = {
            "parties": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
            "dates": [ent.text for ent in doc.ents if ent.label_ == "DATE"],
            "clauses": [
                sentence for sentence in text.split(".")
                if "indemnity" in sentence.lower() or "confidentiality" in sentence.lower()
            ]
        }
        
        # Remove temporary file
        os.remove("temp.pdf")
        return extracted_info, documents
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, None

# Function to Generate Embeddings and Store in FAISS
def process_and_store_embeddings(documents):
    """Generate embeddings and store in FAISS."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        texts = [chunk.page_content for chunk in chunks]
        vector_store = FAISS.from_texts(texts, embedding_model)
        return vector_store
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

# Function to Create a Q&A Chain
def create_qa_chain(vector_store):
    """Create a Q&A chain with GPT-3.5 and FAISS retriever."""
    try:
        llm = OpenAI(temperature=0.0)
        retriever = vector_store.as_retriever()
        qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
        return qa_chain
    except Exception as e:
        st.error(f"Error creating Q&A chain: {str(e)}")
        return None

# Streamlit UI
st.title("PDF Document Analysis and Q&A")

# API Key Input
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# File upload
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None and api_key:
    # Process the uploaded file
    if st.session_state.vector_store is None:
        with st.spinner('Processing document...'):
            # Extract information and create vector store
            extracted_info, documents = parse_and_extract(uploaded_file)
            if documents is not None:
                st.session_state.extracted_info = extracted_info
                st.session_state.vector_store = process_and_store_embeddings(documents)
                if st.session_state.vector_store is not None:
                    st.session_state.qa_chain = create_qa_chain(st.session_state.vector_store)
        
        # Display extracted information
        if st.session_state.extracted_info:
            st.subheader("Extracted Information")
            st.write("Parties:", st.session_state.extracted_info["parties"])
            st.write("Dates:", st.session_state.extracted_info["dates"])
            st.write("Key Clauses:", st.session_state.extracted_info["clauses"])

    # Q&A Interface
    if st.session_state.qa_chain:
        st.subheader("Ask Questions")
        question = st.text_input("Enter your question:")
        
        if question:
            with st.spinner('Generating answer...'):
                try:
                    result = st.session_state.qa_chain({"question": question, "chat_history": st.session_state.chat_history})
                    answer = result["answer"]
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
                    
                    # Display chat history
                    st.subheader("Chat History")
                    for q, a in st.session_state.chat_history:
                        st.write(f"Q: {q}")
                        st.write(f"A: {a}")
                        st.write("---")
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

elif not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
else:
    st.info("Please upload a PDF document to begin.")

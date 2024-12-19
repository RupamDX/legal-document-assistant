import streamlit as st
import spacy
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

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'extracted_info' not in st.session_state:
    st.session_state.extracted_info = None

# Load Models
@st.cache_resource
def load_models():
    nlp = spacy.load('en_core_web_sm')
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return nlp, embedding_model

nlp, embedding_model = load_models()

# Function to Parse and Extract Key Information from PDFs
def parse_and_extract(uploaded_file):
    """Parse and extract key information from a PDF."""
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Load the PDF document
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Extract text from the document
    text = " ".join([doc.page_content for doc in documents])

    # Use spaCy for Named Entity Recognition (NER)
    doc = nlp(text)
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

# Function to Generate Embeddings and Store in FAISS
def process_and_store_embeddings(documents):
    """Generate embeddings and store in FAISS."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]
    vector_store = FAISS.from_texts(texts, embedding_model)
    return vector_store

# Function to Create a Q&A Chain
def create_qa_chain(vector_store):
    """Create a Q&A chain with GPT-3.5 and FAISS retriever."""
    llm = OpenAI(temperature=0.0)
    retriever = vector_store.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    return qa_chain

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
            st.session_state.extracted_info, documents = parse_and_extract(uploaded_file)
            st.session_state.vector_store = process_and_store_embeddings(documents)
            st.session_state.qa_chain = create_qa_chain(st.session_state.vector_store)
        
        # Display extracted information
        st.subheader("Extracted Information")
        st.write("Parties:", st.session_state.extracted_info["parties"])
        st.write("Dates:", st.session_state.extracted_info["dates"])
        st.write("Key Clauses:", st.session_state.extracted_info["clauses"])

    # Q&A Interface
    st.subheader("Ask Questions")
    question = st.text_input("Enter your question:")
    
    if question:
        with st.spinner('Generating answer...'):
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

elif not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
else:
    st.info("Please upload a PDF document to begin.")

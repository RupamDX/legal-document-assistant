import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy
import os
import spacy
import subprocess


# Ensure spaCy model is installed and loaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# Load SentenceTransformer Embedding Model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.title("📄 Legal Document Assistant")
st.sidebar.title("Upload Your PDF Document")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Step 1: Parse and Extract Key Information
    def parse_and_extract(file):
        """Parse a PDF and extract key information."""
        loader = PyPDFLoader(file)
        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        doc = nlp(text)
        extracted_info = {
            "parties": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
            "dates": [ent.text for ent in doc.ents if ent.label_ == "DATE"],
            "clauses": [
                sentence.strip() for sentence in text.split(".")
                if "indemnity" in sentence.lower() or "confidentiality" in sentence.lower()
            ]
        }
        return extracted_info, documents

    # Process Uploaded File
    with st.spinner("Processing your document..."):
        extracted_info, documents = parse_and_extract(uploaded_file)

    st.subheader("Extracted Information")
    st.write("**Parties:**", extracted_info["parties"])
    st.write("**Dates:**", extracted_info["dates"])
    st.write("**Clauses:**", extracted_info["clauses"])

    # Step 2: Generate Embeddings and Store in FAISS
    def process_and_store_embeddings(documents):
        """Generate embeddings and store in FAISS."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        texts = [chunk.page_content for chunk in chunks]
        vector_store = FAISS.from_texts(texts, embedding_model)
        return vector_store

    with st.spinner("Generating embeddings..."):
        vector_store = process_and_store_embeddings(documents)

    # Step 3: Create a Q&A Chain
    def create_qa_chain(vector_store):
        """Create a Q&A chain with GPT-3.5 Turbo and FAISS retriever."""
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        retriever = vector_store.as_retriever()
        qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
        return qa_chain

    qa_chain = create_qa_chain(vector_store)

    # Step 4: Interactive Q&A
    st.subheader("Ask Questions About Your Document")
    chat_history = []
    user_question = st.text_input("Type your question here and press Enter:")

    if user_question:
        with st.spinner("Finding the answer..."):
            answer = qa_chain({"question": user_question, "chat_history": chat_history})
            st.write("**Answer:**", answer["answer"])
            chat_history.append((user_question, answer["answer"]))

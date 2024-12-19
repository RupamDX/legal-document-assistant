import streamlit as st
import os
import re
from typing import Tuple, Optional, Dict, Any

# Download necessary NLTK data
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def import_required_packages():
    try:
        import nltk
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.embeddings import SentenceTransformerEmbeddings
        from langchain.vectorstores import Chroma
        from langchain_openai import ChatOpenAI
        from langchain.chains import ConversationalRetrievalChain
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        return True
    except ImportError as e:
        st.error(f"Failed to import required packages: {str(e)}")
        return False

# Check if imports are successful
if not import_required_packages():
    st.error("Failed to import required packages. Please check your installation.")
    st.stop()

# Now import all required packages
import nltk
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'extracted_info' not in st.session_state:
    st.session_state.extracted_info = None

@st.cache_resource
def load_embedding_model() -> Optional[SentenceTransformerEmbeddings]:
    """Load the embedding model."""
    try:
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        return embedding_model
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

def extract_entities_and_dates(text: str) -> Dict[str, Any]:
    """Extract organizations and dates using NLTK."""
    # Tokenize and tag the text
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    chunks = nltk.ne_chunk(pos_tags)

    # Extract organizations
    organizations = []
    for subtree in chunks.subtrees():
        if subtree.label() == 'ORGANIZATION':
            org = " ".join(w for w, t in subtree.leaves())
            organizations.append(org)

    # Simple regex-based date extraction (handles some formats)
    date_pattern = r'\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})\b'
    dates = re.findall(date_pattern, text, flags=re.IGNORECASE)

    # Extract key clauses
    clauses = [
        sentence.strip() for sentence in text.split(".")
        if any(keyword in sentence.lower() for keyword in ["indemnity", "confidentiality"])
    ]

    return {
        "parties": organizations,
        "dates": dates,
        "clauses": clauses
    }

def parse_and_extract(uploaded_file) -> Tuple[Optional[Dict[str, Any]], Optional[list]]:
    """Parse and extract key information from a PDF using NLTK."""
    try:
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load the PDF document
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # Extract text from the document
        text = " ".join([doc.page_content for doc in documents])

        extracted_info = extract_entities_and_dates(text)

        # Remove temporary file
        os.remove("temp.pdf")
        return extracted_info, documents
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, None

def process_and_store_embeddings(documents: list, embedding_model: SentenceTransformerEmbeddings) -> Optional[Chroma]:
    """Generate embeddings and store in FAISS."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        texts = [chunk.page_content for chunk in chunks]
        return Chroma.from_texts(texts, embedding_model)
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

def create_qa_chain(vector_store: Chroma) -> Optional[ConversationalRetrievalChain]:
    """Create a Q&A chain with GPT-3.5 and FAISS retriever."""
    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Error creating Q&A chain: {str(e)}")
        return None

# Streamlit UI
st.title("PDF Document Analysis and Q&A")

# API Key handling
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Initialize embedding model
if 'embedding_model' not in st.session_state:
    with st.spinner('Loading embedding model...'):
        embedding_model = load_embedding_model()
        if embedding_model is None:
            st.error("Failed to load embedding model.")
            st.stop()
        st.session_state.embedding_model = embedding_model

# File upload
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None and api_key:
    # Process the uploaded file
    if st.session_state.vector_store is None:
        with st.spinner('Processing document...'):
            extracted_info, documents = parse_and_extract(uploaded_file)
            if documents is not None:
                st.session_state.extracted_info = extracted_info
                st.session_state.vector_store = process_and_store_embeddings(
                    documents,
                    st.session_state.embedding_model
                )
                if st.session_state.vector_store is not None:
                    st.session_state.qa_chain = create_qa_chain(st.session_state.vector_store)
        
        # Display extracted information
        if st.session_state.extracted_info:
            st.subheader("Extracted Information")
            st.write("Parties:", st.session_state.extracted_info["parties"])
            st.write("Dates:", st.session_state.extracted_info["dates"])
            if st.session_state.extracted_info["clauses"]:
                st.write("Key Clauses:", st.session_state.extracted_info["clauses"])

    # Q&A Interface
    if st.session_state.qa_chain:
        st.subheader("Ask Questions")
        question = st.text_input("Enter your question:")
        
        if question:
            with st.spinner('Generating answer...'):
                try:
                    result = st.session_state.qa_chain({
                        "question": question,
                        "chat_history": st.session_state.chat_history
                    })
                    answer = result["answer"]
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
                    
                    # Display chat history
                    st.subheader("Chat History")
                    for q, a in st.session_state.chat_history:
                        st.write("Question:", q)
                        st.write("Answer:", a)
                        st.write("---")
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

elif not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to continue.")
else:
    st.info("Please upload a PDF document to begin.")

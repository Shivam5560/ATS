import streamlit as st
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from fuzzywuzzy import fuzz
import numpy as np
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
import tempfile
from dotenv import load_dotenv
import os

# NLTK downloads and initializations
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return " ".join([lemmatizer.lemmatize(token) for token in tokens])

# Key phrase extraction function
def extract_key_phrases(text):
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    noun_phrases = []
    current_phrase = []
    
    for word, tag in pos_tags:
        if tag.startswith('NN') or tag.startswith('JJ'):
            current_phrase.append(word)
        elif current_phrase:
            noun_phrases.append(' '.join(current_phrase))
            current_phrase = []
    
    if current_phrase:
        noun_phrases.append(' '.join(current_phrase))
    
    return noun_phrases

# Keyword matching function
def keyword_matching(resume_text, job_desc):
    resume_keywords = set(extract_key_phrases(resume_text))
    job_keywords = set(extract_key_phrases(job_desc))
    
    matched = resume_keywords.intersection(job_keywords)
    return len(matched) / len(job_keywords) if job_keywords else 0

# Fuzzy matching function
def fuzzy_match_score(resume_text, job_desc):
    return fuzz.token_set_ratio(resume_text, job_desc) / 100

# TF-IDF similarity function
def tfidf_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Section score calculation function
def calculate_section_score(section, job_desc):
    preprocessed_section = preprocess_text(" ".join(section))
    preprocessed_job_desc = preprocess_text(job_desc)
    
    keyword_score = keyword_matching(preprocessed_section, preprocessed_job_desc)
    fuzzy_score = fuzzy_match_score(preprocessed_section, preprocessed_job_desc)
    tfidf_score = tfidf_similarity(preprocessed_section, preprocessed_job_desc)
    
    base_score = 0.5
    weighted_score = (keyword_score * 0.4 + fuzzy_score * 0.3 + tfidf_score * 0.3) * 0.5
    
    return base_score + weighted_score

# Advanced ATS similarity score function
def advanced_ats_similarity_score(resume_dict, job_description):
    work_exp = " ".join([f"{exp['job_title']} {exp['company']} {' '.join(exp['responsibilities'])}" 
                         for exp in resume_dict["work_experience"]])
    projects = " ".join([f"{proj['name']} {proj['description']}" for proj in resume_dict["projects"]])
    skills = " ".join(resume_dict["skills"])
    certifications = " ".join(resume_dict["certifications"])

    work_exp_score = calculate_section_score([work_exp], job_description)
    projects_score = calculate_section_score([projects], job_description)
    skills_score = calculate_section_score([skills], job_description)
    cert_score = calculate_section_score([certifications], job_description)

    weights = [0.40, 0.30, 0.25, 0.05]
    final_score = np.average([work_exp_score, projects_score, skills_score, cert_score], weights=weights)

    curved_score = min(1.0, final_score * 1.2)
    
    score = round(curved_score * 100, 2)
    is_accepted = score >= 65

    return {
        "similarity_score": score,
        "is_accepted": is_accepted,
        "section_scores": {
            "work_experience": round(work_exp_score * 100, 2),
            "projects": round(projects_score * 100, 2),
            "skills": round(skills_score * 100, 2),
            "certifications": round(cert_score * 100, 2)
        }
    }

# Load environment variables
load_dotenv()
HF_KEY = os.getenv("HF_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# File saving function
def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    return file_paths

# Model generation function
def generate_model(file_paths):
    try:
        if not file_paths:
            st.error("Please upload files to proceed.")
            return None
        
        with st.spinner("Processing documents..."):
            reader = SimpleDirectoryReader(input_files=file_paths)
            documents = reader.load_data()
            text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
            nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)
            
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", token=HF_KEY)
            llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
            
            Settings.embed_model = embed_model
            Settings.llm = llm
            
            vector_index = VectorStoreIndex.from_documents(documents, show_progress=True, node_parser=nodes)
            vector_index.storage_context.persist(persist_dir="./storage_mini")
            
            storage_context = StorageContext.from_defaults(persist_dir="./storage_mini")
            index = load_index_from_storage(storage_context)
            
        st.success("Documents processed successfully!")
        return index.as_query_engine(similarity_top_k=2)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Main function
def main():
    st.set_page_config(page_title="BRAINY BUDDY", page_icon="🧠", layout="wide")

    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #F0F4F8;
        color: #2C3E50;
        font-family: 'Roboto', sans-serif;
    }
    .stButton > button {
        background-color: #3498DB;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2980B9;
    }
    .stTextInput > div > div > input {
        background-color: #ECF0F1;
        color: #2C3E50;
        border-radius: 5px;
    }
    .stProgress > div > div > div > div {
        background-color: #3498DB;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧠 BRAINY BUDDY")
    st.markdown("#### Your AI-powered Resume Analyzer")

    if "file_paths" not in st.session_state:
        st.session_state.file_paths = None
    if "model" not in st.session_state:
        st.session_state.model = None

    with st.sidebar:
        st.header("📁 Upload Files")
        uploaded_files = st.file_uploader("", type=['pdf'], accept_multiple_files=True)
        if uploaded_files:
            file_paths = save_uploaded_files(uploaded_files)
            if file_paths != st.session_state.file_paths:
                st.session_state.file_paths = file_paths
                st.session_state.model = generate_model(file_paths)

    if st.session_state.model:
        st.header("💼 Job Description")
        job_description = st.text_area("Enter the job description here:", height=200)
        
        if st.button("Analyze Resume"):
            if job_description:
                with st.spinner("Analyzing resume..."):
                    response = st.session_state.model.query("Extract key information from the resume and format it as a dictionary.").response
                    try:
                        resume_dict = eval(response)
                        score = advanced_ats_similarity_score(resume_dict, job_description)
                        
                        st.header("📊 Analysis Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = go.Figure(data=[go.Pie(values=[score['similarity_score'], 100-score['similarity_score']], 
                                         hole=.3, 
                                         marker_colors=['#3498DB', '#ECF0F1'])])
                            fig.update_layout(
                                annotations=[dict(text=f"{score['similarity_score']}%", x=0.5, y=0.5, font_size=20, showarrow=False)],
                                margin=dict(t=0, b=0, l=0, r=0),
                                height=300
                            )
                            st.plotly_chart(fig)
                        
                        with col2:
                            st.subheader("Match Status")
                            status = "Accepted" if score['is_accepted'] else "Not Accepted"
                            color = "green" if score['is_accepted'] else "red"
                            st.markdown(f"<h2 style='text-align: center; color: {color};'>{status}</h2>", unsafe_allow_html=True)
                        
                        st.header("📈 Section Scores")
                        for section, section_score in score['section_scores'].items():
                            st.subheader(f"{section.replace('_', ' ').title()}")
                            st.progress(section_score / 100)
                            st.write(f"{section_score}%")
                        
                        with st.expander("View Extracted Resume Details"):
                            st.json(resume_dict)
                    
                    except Exception as e:
                        st.error(f"Error processing resume: {e}")
            else:
                st.warning("Please enter a job description.")
    else:
        st.info("Please upload resume files to begin analysis.")

if __name__ == "__main__":
    main()

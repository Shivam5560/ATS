# app.py
import streamlit as st
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
import tempfile
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
from transformers import DistilBertTokenizer, DistilBertModel

def dict_to_string(data):
    result = []

    # Personal Info
    personal_info = data.get('personal_info', {})
    result.append(f"{personal_info.get('name', '')}")
    result.append(f"{personal_info.get('email', '')}")
    result.append(f"{personal_info.get('phone', '')}")
    result.append("")

    # Education
    result.append("EDUCATION")
    for edu in data.get('education', []):
        result.append(f"{edu.get('degree', '')}")
        result.append(f"{edu.get('institution', '')}")
        result.append(f"{edu.get('graduation_date', '')}")
    result.append("")

    # Work Experience
    result.append("WORK EXPERIENCE")
    for exp in data.get('work_experience', []):
        result.append(f"{exp.get('job_title', '')}")
        result.append(f"{exp.get('company', '')}")
        result.append(f"{exp.get('dates', '')}")
        for resp in exp.get('responsibilities', []):
            result.append(f"- {resp}")
    result.append("")

    # Skills
    result.append("SKILLS")
    result.append(", ".join(data.get('skills', [])))
    result.append("")

    # Projects
    result.append("PROJECTS")
    for project in data.get('projects', []):
        result.append(f"{project.get('name', '')}")
        result.append(f"{project.get('description', '')}")
    result.append("")

    # Certifications
    result.append("CERTIFICATIONS")
    for cert in data.get('certifications', []):
        result.append(cert)

    return "\n".join(result).strip()

def advanced_ats_similarity_score(dictionary,jd):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    dictionary_str =  dict_to_string(dictionary)

    # Tokenize and embed job descriptions
    job_description_embeddings = []
    tokens = tokenizer(jd, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**tokens)
    embeddings = output.last_hidden_state.mean(dim=1).numpy()
    job_description_embeddings.append(embeddings[0])

    # Tokenize and embed job descriptions
    resume_embeddings = []
    tokens = tokenizer(dictionary_str, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**tokens)
    embeddings = output.last_hidden_state.mean(dim=1).numpy()
    resume_embeddings.append(embeddings[0])  # Flatten the embeddings to 1D

    similarity_scores = cosine_similarity(job_description_embeddings, resume_embeddings)
    print(similarity_scores)
    return similarity_scores[0]







#from llama_index.embeddings.mistralai import MistralAIEmbedding
template = """You are an AI assistant trained to extract key information from resumes. Your task is to analyze the given resume text and extract relevant details into a structured dictionary format. Please follow these guidelines:

1. Read the entire resume carefully and extract all the subheaders with all the details in the following format , also do not change the headers subdata the order should be same.
2. Extract the following information:
   - Personal Information (name, email, phone number)
   - Education (degrees, institutions, graduation dates)
   - Work Experience or Professional Expereinces(job titles, companies, dates, key responsibilities)
   - Skills
   - Projects (if any)
   - Certifications (if any)

3. Organize the extracted information into a dictionary with the following structure:

{
    "personal_info": {
        "name": "",
        "email": "",
        "phone": ""
    },
    "education": [
        {
            "degree": "",
            "institution": "",
            "graduation_date": ""
        }
    ],
    "work_experience": [
        {
            "job_title": "",
            "company": "",
            "dates": "",
            "responsibilities": []
        }
    ],
    "skills": [],
    "projects": [
        {
            "name": "",
            "description": ""
        }
    ],
    "certifications": []
}

4. Fill in the dictionary with the extracted information and in correct order also from the resume by cross checking with their headers and the exracted value.
5. If any section is not present in the resume, leave it as an empty list or dictionary as appropriate.
6. Ensure all extracted information is accurate and relevant.
7. Return the completed dictionary.
8. Match the dictionary key values with the resume subheaders like personal info and all and do the needfull.

Resume text:
[Insert resume text here]

Please provide the extracted information in the specified dictionary format.")"""



load_dotenv()
HF_KEY  =  os.getenv("HF_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    return file_paths

def generate_model(file_paths):
    global template
    try:
        if not file_paths:
            st.error("Please upload files to proceed.")
            return None
        st.info("Loading and processing documents...")
        reader = SimpleDirectoryReader(input_files=file_paths)
        documents = reader.load_data()
        st.info("Splitting text into nodes...")
        text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
        nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)
        st.info("Initializing embedding model and language model...")
        # embed_model = MistralAIEmbedding(api_key=MISTRAL_API_KEY)
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2",token=HF_KEY)
        llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
        st.info("Creating service context...")
        # service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
        Settings.embed_model = embed_model
        Settings.llm = llm
        st.info("Creating vector index from documents...")
        vector_index = VectorStoreIndex.from_documents(documents, show_progress=True,node_parser=nodes)# service_context=service_context, 
        vector_index.storage_context.persist(persist_dir="./storage_mini")
        st.info("Loading index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir="./storage_mini")
        index = load_index_from_storage(storage_context,) #service_context=service_context)
        st.success("PDF loaded successfully!")
        chat_engine = index.as_query_engine(similarity_top_k=2,BasePromptTemplate=template)
        return chat_engine
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def main():
    st.set_page_config(page_title="KHOJO BHAI", page_icon="📝", layout="wide")

    # Custom CSS for improved UI
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
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #ECF0F1;
        color: #2C3E50;
        border-radius: 5px;
    }
    .stProgress > div > div > div > div {
        background-color: #3498DB;
    }
    .sidebar .sidebar-content {
        background-color: #34495E;
        color: #ECF0F1;
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    .stAlert {
        background-color: #E74C3C;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("📝 KHOJO BHAI")
    st.markdown("#### Your AI-powered Resume Analyzer")

    if "file_paths" not in st.session_state:
        st.session_state.file_paths = None
    if "model" not in st.session_state:
        st.session_state.model = None

    with st.sidebar:
        st.header("📁 Upload Files")
        uploaded_files = st.file_uploader("", type=['pdf'], accept_multiple_files=True)
        if uploaded_files:
            with st.spinner("Processing files..."):
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
                    response = st.session_state.model.query(template).response
                    st.markdown(response)
                    try:
                        resume_dict = eval(response)
                        score = advanced_ats_similarity_score(resume_dict, job_description)
                        
                        st.header("📊 Analysis Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = go.Figure(data=[go.Pie(values=[score['similarity_score'], 100-score['similarity_score']], 
                                         hole=.3, 
                                         marker_colors=['#3498DB', '#ECF0F1'],
                                         textinfo='none')])
                            fig.update_layout(
                                annotations=[dict(text=f"{score['similarity_score']}%", x=0.5, y=0.5, font_size=20, showarrow=False)],
                                margin=dict(t=0, b=0, l=0, r=0),
                                height=300,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)'
                            )
                            st.plotly_chart(fig)
                        
                        with col2:
                            st.subheader("Match Status")
                            status = "Accepted" if score['is_accepted'] else "Not Accepted"
                            color = "#2ECC71" if score['is_accepted'] else "#E74C3C"
                            st.markdown(f"<h2 style='text-align: center; color: {color};'>{status}</h2>", unsafe_allow_html=True)
                        
                        st.header("📈 Section Scores")
                        for section, section_score in score['section_scores'].items():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.subheader(f"{section.replace('_', ' ').title()}")
                                st.progress(section_score / 100)
                            with col2:
                                st.markdown(f"<h3 style='text-align: center; margin-top: 20px;'>{section_score}%</h3>", unsafe_allow_html=True)
                        
                        with st.expander("View Extracted Resume Details"):
                            st.json(resume_dict)
                    
                    except Exception as e:
                        st.error(f"Error processing resume , the text could not be extacted properly from your resume, do check your fonts and format properly.")
            else:
                st.warning("Please enter a job description.")
    else:
        st.info("Please upload resume files to begin analysis.")

if __name__ == "__main__":
    main()

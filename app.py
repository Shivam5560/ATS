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

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from fuzzywuzzy import fuzz
import numpy as np

def safe_nltk_download(package):
    try:
        nltk.download(package, quiet=True)
    except Exception as e:
        print(f"Failed to download {package}: {e}")

safe_nltk_download('punkt')
safe_nltk_download('stopwords')
safe_nltk_download('averaged_perceptron_tagger')
safe_nltk_download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    except LookupError:
        pass
    lemmatized = []
    for token in tokens:
        try:
            lemmatized.append(lemmatizer.lemmatize(token))
        except LookupError:
            lemmatized.append(token)
    return " ".join(lemmatized)

def extract_key_phrases(text):
    try:
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
    except LookupError:
        tokens = text.split()
        pos_tags = [(token, 'NN') for token in tokens]
    
    noun_phrases = []
    current_phrase = []
    
    for word, tag in pos_tags:
        if tag.startswith('NN') or tag.startswith('JJ'):  # Include adjectives
            current_phrase.append(word)
        elif current_phrase:
            noun_phrases.append(' '.join(current_phrase))
            current_phrase = []
    
    if current_phrase:
        noun_phrases.append(' '.join(current_phrase))
    
    return noun_phrases

def keyword_matching(resume_text, job_desc):
    resume_keywords = set(extract_key_phrases(resume_text))
    job_keywords = set(extract_key_phrases(job_desc))
    
    matched = resume_keywords.intersection(job_keywords)
    return len(matched) / len(job_keywords) if job_keywords else 0

def fuzzy_match_score(resume_text, job_desc):
    return fuzz.token_set_ratio(resume_text, job_desc) / 100

def tfidf_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def calculate_section_score(section, job_desc):
    preprocessed_section = preprocess_text(" ".join(section))
    preprocessed_job_desc = preprocess_text(job_desc)
    
    keyword_score = keyword_matching(preprocessed_section, preprocessed_job_desc)
    fuzzy_score = fuzzy_match_score(preprocessed_section, preprocessed_job_desc)
    tfidf_score = tfidf_similarity(preprocessed_section, preprocessed_job_desc)
    
    # Increase the base score and adjust the weights
    base_score = 0.5  # Start with a base score of 0.5
    weighted_score = (keyword_score * 0.4 + fuzzy_score * 0.3 + tfidf_score * 0.3) * 0.5
    
    return base_score + weighted_score

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

    # Adjust weights to favor skills and work experience
    weights = [0.35, 0.25, 0.3, 0.1]  # work_exp, projects, skills, certifications
    final_score = np.average([work_exp_score, projects_score, skills_score, cert_score], weights=weights)

    # Apply a more lenient scoring curve
    curved_score = min(1.0, final_score * 1.2)  # Increase scores by up to 20%
    
    score = round(curved_score * 100, 2)
    is_accepted = score >= 65  # Lower the acceptance threshold

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


#from llama_index.embeddings.mistralai import MistralAIEmbedding
template = """You are an AI assistant trained to extract key information from resumes. Your task is to analyze the given resume text and extract relevant details into a structured dictionary format. Please follow these guidelines:

1. Read the entire resume carefully.
2. Extract the following information:
   - Personal Information (name, email, phone number)
   - Education (degrees, institutions, graduation dates)
   - Work Experience (job titles, companies, dates, key responsibilities)
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

4. Fill in the dictionary with the extracted information.
5. If any section is not present in the resume, leave it as an empty list or dictionary as appropriate.
6. Ensure all extracted information is accurate and relevant.
7. Return the completed dictionary.

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
    global template
    st.set_page_config(
        page_title="RAG Based Chatbot",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #FFFFFF;
    }
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #B5FFE9; /* Body theme color */
    }
    
    .stApp {
        font-family:'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;
        background-color: #C5E0D8; /* Body theme color */
    }
    
    # .element-container:has(>.stTextArea), .stTextArea {
    #     width: 800px !important;
    # }
    .stTextArea textarea {
        height: 400px; 
        width : 1200px;
    }
    
    .stMain {
        padding: 20px;
    }
    
    .stTextInput {
        border: none; /* Remove black line border */
        padding: 10px;
        border-radius: 4px;
        color: #4B0082; /* Indigo */
    }
    
    .stTextInput:focus {
        outline: none;
        box-shadow: 0 0 5px #8A2BE2; /* BlueViolet */
    }
    
    .stButton:not([type="submit"]) {
        background-color: #B3CDD1; 
        max-width: fit-content;
        color: #B3CDD1;
    }
</style>
""", unsafe_allow_html=True)
    st.markdown("""
    # Welcome to BRAINY BUDDY! 
    ###### This chatbot utilizes a Retrieval-Augmented Generation (RAG) model to provide accurate and relevant responses based on the information contained in the uploaded files.

    
    """)
    st.sidebar.title("BRAINY BUDDY")

    if "file_paths" not in st.session_state:
        st.session_state.file_paths = None
    if "model" not in st.session_state:
        st.session_state.model = None

    with st.sidebar:
        uploaded_files = st.file_uploader(label='',type=['pdf'], accept_multiple_files=True)
        if uploaded_files:
            file_paths = save_uploaded_files(uploaded_files)
            if file_paths != st.session_state.file_paths:
                st.session_state.file_paths = file_paths
                st.session_state.model = generate_model(file_paths)

    if st.session_state.model:
        #user_input = st.text_input("Question", key="question_input", placeholder="Ask your question here...", label_visibility="collapsed")
        user_input = template
        jobd = st.text_input("Job Description", key="jobd", placeholder="JoB Description.", label_visibility="collapsed")
        _,col2,_ = st.columns(3)
        with col2:
            button = st.button(label='Enter',type='primary')
        if button:
            with st.spinner():
                response = st.session_state.model.query(user_input).response
                #st.text_area("Response", value=response)
                # st.markdown(f"**Formatted Response:**\n{str(response)}")
                if isinstance(response, str):
                    try:
                        response = eval(response)  # This converts the string to a dictionary if it's in the correct format
                    except:
                        st.error("Error: Response is not in the correct format")
                        return

                if not isinstance(response, dict):
                    st.error(f"Error: Response is not a dictionary. Type: {type(response)}")
                    st.write(response)  # This will display the content of response
                    return

                score = advanced_ats_similarity_score(response, jobd)
                st.markdown(score)

    else:
        st.info("Please upload files to initiate the chatbot.")

if __name__ == "__main__":
    main()

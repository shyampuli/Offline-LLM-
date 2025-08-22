import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import (                                                                            
    SystemMessagePromptTemplate,                                                                            
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

import requests
import json
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstore import MemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompt import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for Model Selection and Navigation
st.sidebar.title("⚙ Configuration")
selected_model = st.sidebar.selectbox(
    "Choose Model",
    ["deepseek-r1:1.5b", "deepseek-r1:3b"],
    index=0
)

sidebar_option = st.sidebar.selectbox(
    "Choose an option:",
    ["Ollama Offline LLM", "Document Summarizer"]
)

# Define the API URL for Ollama

def generate_response(prompt):
    data = {
        "model": selected_model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(data))

    if response.status_code == 200:
        data = json.loads(response.text)
        return data['response']
    else:
        return "Error processing request."


# Document Summarizer Section
if sidebar_option == "Document Summarizer":
    st.title("📘 Offline AI Document Analyzer")
    st.markdown("### Your Personal AI Sidekick for Document Analysis")
    st.markdown("---")

    PDF_STORAGE_PATH = 'document_store/pdfs/'
    EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
    DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
    LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

    def save_uploaded_file(uploaded_file):
        file_path = PDF_STORAGE_PATH + uploaded_file.name
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        return file_path

    def load_pdf_documents(file_path):
        document_loader = PDFPlumberLoader(file_path)
        return document_loader.load()

    def chunk_documents(raw_documents):
        text_processor = RecursiveCharacterTextSplitter(
            chunk_size=1000,            
            add_start_index=True
        )
        return text_processor.split_documents(raw_documents)

    def find_related_documents(query):
        return DOCUMENT_VECTOR_DB.similarity_search(query)

    def generate_answer(user_query, context_documents):
        
        conversation_prompt = ChatPromptTemplate.from_template("""
        You are an expert research assistant. Use the provided context to answer the query.
        If unsure, state that you don't know. Be concise and factual (max 3 sentences).

        Query: {user_query}
        Context: {document_context}
        Answer:
        """)
        response_chain = conversation_prompt | LANGUAGE_MODEL
        return response_chain.invoke({"user_query": user_query, "document_context": context_text})

    # File Upload Section
    uploaded_pdf = st.file_uploader("Upload Document (PDF)", type="pdf", help="Select a PDF document for analysis", accept_multiple_files=False)

    if uploaded_pdf:
        saved_path = save_uploaded_file(uploaded_pdf)
        raw_docs = load_pdf_documents(saved_path)
        index_documents(processed_chunks)

        st.success("✅ Document processed successfully! Ask your questions below.")

        user_input = st.chat_input("Enter your question about the document...")

        if user_input:
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("🧠 Processing..."):
                relevant_docs = find_related_documents(user_input)
                ai_response = generate_answer(user_input, relevant_docs)

            with st.chat_message("assistant", avatar="🤖"):
                st.write(ai_response)

# Ollama Offline LLM Section
elif sidebar_option == "Ollama Offline LLM":
    st.title("🌟 Ollama Offline LLM: Your Personal AI Sidekick 🌟")
    history = []

    def generate_response(prompt):
        history.append(prompt)
        final_prompt = '\n'.join(history)
        data = {
            "model": selected_model,
            "prompt": final_prompt,
            "stream": False
        }

        

        if response.status_code == 200:
            data = json.loads(response.text)
            return data['response']
        else:
            return "Error processing request."

    if "message_log" not in session_state:
        st.session_state.message_log = [
            {"role": "ai", "content": "Hi there! I'm your friendly AI assistant. How can I help you today? 😊"}]

    chat_container = st.container()

    with chat_container:
        for message in st.session_state.message_log:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_query = st.cht_input("Ask me anything...")

    if user_query:
        st.session_state.message_log.append({"role": "user", "content": user_query})
        with st.spinner("🧠 Processing..."):
            ai_response = generate_response(user_query)
     
        st.rerun()

# 🦙 Offline LLM with Streamlit + Ollama

This project lets you run an **offline AI assistant** using [Ollama](https://ollama.ai) and [LangChain](https://python.langchain.com).  
It has two modes:
- **Ollama Offline LLM** – chat with a local model.
- **Document Summarizer** – upload PDFs and ask AI questions.

---

## 🚀 Setup

1. Install [Ollama](https://ollama.ai/download) and pull a model (example):
   ```bash
   ollama pull deepseek-r1:1.5b
   
2. Clone this repo and install dependencies:

git clone <your-repo-url>
cd <repo>
pip install -r requirements.txt

3. Start Ollama server (runs on http://localhost:11434 by default):

ollama server

4.Run the Streamlit app:

streamlit run app.py

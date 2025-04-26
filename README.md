# Simple RAG Application with Ollama and Langgraph

Ollama-RAG is a local LLM chatbot that uses LangGraph's workflow to decide when to retrieve external context or rely on the model’s internal knowledge, depending on the user's query. The chatbot is deployed with a Streamlit UI for easy interaction.

👉 Want to learn how it works step-by-step? Check out my Medium article: [Simple RAG Application with Ollama](https://medium.com/@vdcapriles/simple-rag-application-with-ollama-and-langgraph-259450772903).

## 🧩 How does the application works?

The app uses LangGraph to create a branching workflow. Depending on the input, it either:

- Calls a retriever to fetch relevant documents

- Or directly responds using the model’s internal knowledge

<div align="left">
  <img src="https://github.com/victor-capriles/ollama-rag/blob/main/graph_wf.png" width="30%" alt="langgraph wf" />
</div>

## 📦 Installation

You can either install the dependencies for this project using:

Option 1 - pip

```python
pip install -r requirements.txt
```
Option 2 - poetry
```python
pip install poetry # if you don't have it already installed
poetry install
```

## 🚀 Running the App

To launch the chatbot, run the following command in your terminal:

```python
streamlit run main.py
```



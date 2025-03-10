import os
import asyncio
if os.name == "nt":  # Windows-specific fix
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langgraph.graph import END, START
from langgraph.prebuilt import ToolNode, tools_condition


import streamlit as st

# streamlit app set-up
st.set_page_config(page_title="RAG Application", layout="wide")
st.title("Chat with Bart 🤖")

# declare the model
llm = ChatOllama(
    model="llama3.2",
    temperature=0.1
)

# initialize the model using HuggingFaceEmbeddings wrapper
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

# load and process documents
pdf_folder_path = "./pdf"
loader = DirectoryLoader(pdf_folder_path, loader_cls=PyPDFLoader)
documents = loader.load()

# split documents into chunks
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=51
)

# chunked documents
chunked_documents = recursive_splitter.split_documents(documents)

# create chroma vector database from documents
vector_store = Chroma.from_documents(
    documents=chunked_documents,  # chunks we created earlier
    embedding=embedding_model, # embeddings we initialized earlier
    persist_directory="chromadb"
)

# define retrieval tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve factual information from the knowledge base.

    Always provide a query related to Victor Capriles' work experience, education, or background.
    If the query is unclear, ask the user for clarification instead of calling this tool with empty parameters.

    Example inputs:
    - "What did Victor do at HP?"
    - "Where did Victor study?"

    Do NOT call this tool with an empty query."""

    retrieved_docs = vector_store.similarity_search(query, k=8)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# define graph nodes
# generate an aimessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

# execute the retrieval.
tools = ToolNode([retrieve])

# generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # format system prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are a refined and courteous butler. Your responses are always polite and respectful.\n"
        "You must follow these rules:\n"
        "1. If the user's question relates to Victor Capriles or his work, use the provided context.\n"
        "2. If the question is unrelated, ignore the context and answer based on your knowledge.\n"
        "\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

# build graph
graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.add_edge(START, "query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile()

# streamlit chat ui
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# user input
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    response_message = ""
    for output in graph.stream({"messages": st.session_state.messages}, stream_mode="values"):
        for key, value in output.items():
            if isinstance(value, list):
                for message in value:
                    if isinstance(message, AIMessage) and message.content.strip():
                        response_message = message.content

    # Append response and display it
    st.session_state.messages.append(AIMessage(content=response_message))
    with st.chat_message("assistant"):
        st.markdown(response_message)
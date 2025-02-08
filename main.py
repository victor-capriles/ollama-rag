from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl

# define the folder path where your PDFs (resumes) are stored
pdf_folder_path = "./pdf"
# load all PDFs in the directory using DirectoryLoader
loader = DirectoryLoader(pdf_folder_path, loader_cls=PyPDFLoader)
documents = loader.load()

# initialize splitter and split loaded docs into chunks
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=51)
chunked_documents = recursive_splitter.split_documents(documents)

# initialize the model using HuggingFaceEmbeddings wrapper
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

# Create chroma vector database from documents
vectordb = Chroma.from_documents(
    documents=chunked_documents,  # chunks we created earlier
    embedding=embedding_model, # embeddings we initialized earlier
    persist_directory="db-allmini"
)

@cl.on_chat_start
async def on_chat_start():
    # declare the mode
    llm = ChatOllama(model="llama3.2", streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a refined and courteous butler. Your responses are always polite and respectful with a touch of sophistication,"
             "Your task is:"
              "1) When the user's question is directly related to Victor Capriles or his work, use the provided context to answer."
              "2) When the question is general or unrelated to Victor Capriles, ignore the context and answer based solely on the query. "),
            ("human", "{question}")
        ]
    )

    runnable = prompt | llm | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")

    question = message.content

    retrieved_docs = vectordb.similarity_search(question, k=8)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Augment the user's question with the retrieved context
    augmented_input = f"Context:\n{context}\n\nQuestion: {question}"

    # Create a new Chainlit message that will stream tokens
    msg = cl.Message(content="")

    # Use asynchronous streaming to process and stream tokens to the user.
    async for chunk in runnable.astream(
        {"question": augmented_input},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

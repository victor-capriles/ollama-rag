from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage
)
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chainlit as cl
from sympy.polys.polyconfig import query
from urllib3.contrib.emscripten.fetch import streaming_ready

@cl.on_chat_start
async def start():
    llm = Ollama(model="llama2", streaming=True)
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    Settings.llm = llm
    Settings.embed_model = embed_model

    # read documents
    reader = SimpleDirectoryReader('./pdf')
    documents = reader.load_data()

    # split documents into chunks
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=51)
    nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)

    #
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir="./storage")

    #
    query_engine = index.as_query_engine(llm=Settings.llm, embed_model=Settings.embed_model, streaming=True,
                                         similarity_top_k=5)
    cl.user_session.set("query_engine", query_engine)

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    msg = cl.Message(content="", author="Assistant")

    # Use combined string as the input for the query | query is processed and nodes are sent to the llm
    res = await cl.make_async(query_engine.query)(message.content)

    # Stream the response tokens to the user
    for token in res.response_gen:
        await msg.stream_token(token)

    await msg.send()








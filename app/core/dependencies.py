from core.config import NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, OLLAMA_HOST
from langchain_ollama import OllamaEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector

def get_graph():
    return Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PASSWORD)

def get_vector_retriever():
    embed = OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_HOST)
    vector_index = Neo4jVector.from_existing_graph(
        embedding=embed,
        search_type="hybrid",
        url=NEO4J_URL,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database="indonesiandb",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    return vector_index.as_retriever()

def get_vector_retriever_en():
    embed = OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_HOST)
    vector_index = Neo4jVector.from_existing_graph(
        embedding=embed,
        search_type="hybrid",
        url=NEO4J_URL,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database="englishdb",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    return vector_index.as_retriever()
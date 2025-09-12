import os
import json
import requests
import logging
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from core.config import VLLM_CHAT_MODEL, CHAT_TIMEOUT, VLLM_CHAT_URL, VLLM_EMBED_MODEL, EMBED_BATCH, VLLM_EMBEDDINGS_URL, NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLLMEmbeddings(Embeddings):
    def __init__(
        self, 
        base_url: str = VLLM_EMBEDDINGS_URL,
        model: str = VLLM_EMBED_MODEL,
        batch_size: int = EMBED_BATCH
    ):
        self.base_url = base_url
        self.model = model
        self.batch_size = batch_size
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        all_vectors: List[List[float]] = []
        
        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                logger.info(f"Embedding batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1} ({len(batch)} items)")
                
                payload = {
                    "model": self.model,
                    "input": batch,
                    "encoding_format": "float"
                }
                
                response = requests.post(
                    self.base_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=120
                )
                
                if response.status_code != 200:
                    error_msg = f"vLLM API returned {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    raise Exception(f"Embeddings service error: {error_msg}")
                
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse embeddings response: {e}")
                    raise Exception("Invalid response from embeddings service")
                
                embedding_data = data.get("data", [])
                if not embedding_data:
                    raise Exception("No embeddings returned from service")
                
                items = sorted(embedding_data, key=lambda x: x.get("index", 0))
                vectors = [item["embedding"] for item in items]
                
                if len(vectors) != len(batch):
                    error_msg = f"Expected {len(batch)} embeddings, got {len(vectors)}"
                    logger.error(error_msg)
                    raise Exception(f"Embeddings length mismatch: {error_msg}")
                
                all_vectors.extend(vectors)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to embeddings service failed: {e}")
            raise Exception(f"Failed to connect to embeddings service: {str(e)}")
        
        logger.info(f"Successfully generated {len(all_vectors)} embeddings")
        return all_vectors
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_embeddings(texts)
    
    def embed_query(self, text: str) -> List[float]:
        result = self._get_embeddings([text])
        return result[0] if result else []

def get_vllm_chat_llm():
    return ChatOpenAI(
        base_url=VLLM_CHAT_URL,
        api_key="not-needed",
        model=VLLM_CHAT_MODEL,
        temperature=0,
        timeout=CHAT_TIMEOUT,
        max_retries=2
    )

class SafeNeo4jGraph(Neo4jGraph):
    def refresh_schema(self):
        self.structured_schema = {}

def get_graph_upload(language: str):
    database = "englishdb" if language == "english" else "indonesiandb"
    return SafeNeo4jGraph(
        url=NEO4J_URL,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=database
    )

def get_graph():
    return Neo4jGraph(
        url=NEO4J_URL,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
    )

def get_vector_retriever():
    embed = VLLMEmbeddings(
        base_url=VLLM_EMBEDDINGS_URL,
        model=VLLM_EMBED_MODEL,
        batch_size=EMBED_BATCH
    )
    
    vector_index = Neo4jVector.from_existing_graph(
        embedding=embed,
        search_type="hybrid",
        url=NEO4J_URL,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database="indonesiandb",
        node_label="Chunk", 
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    return vector_index.as_retriever()

def get_vector_retriever_en():
    embed = VLLMEmbeddings(
        base_url=VLLM_EMBEDDINGS_URL,
        model=VLLM_EMBED_MODEL,
        batch_size=EMBED_BATCH
    )
    
    vector_index = Neo4jVector.from_existing_graph(
        embedding=embed,
        search_type="hybrid",
        url=NEO4J_URL,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database="englishdb",
        node_label="Chunk",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    return vector_index.as_retriever()
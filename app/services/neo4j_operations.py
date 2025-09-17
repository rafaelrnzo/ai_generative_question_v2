from core.dependencies import get_graph
import logging
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from utils.helpers import is_mcq_request

def delete_data_from_neo4j(filename: str, language: str, graph):
    query = """
    MATCH (n)
    WHERE toLower(n.title) CONTAINS $filename AND toLower(n.language) = $language
    DETACH DELETE n
    RETURN count(n) AS deleted_count
    """
    result = graph.query(query, params={"filename": filename.lower(), "language": language.lower()})
    return result[0]["deleted_count"] if result else 0


def create_nodes_from_text(text: str, filename: str, graph):
    query = """
    CREATE (d:Document {filename: $filename, content: $content})
    """
    graph.run(query, filename=filename, content=text)

def flush_db(graph):
    query = """
    MATCH (n) DETACH DELETE n
    """
    try:
        with graph.session() as session:
            result = session.run(query)
            return f"Successfully deleted {result.consume().counters.nodes_deleted} nodes"
    except Exception as e:
        try:
            result = graph.query(query)  # Your original method
            return "Database successfully flushed"
        except Exception as inner_e:
            try:
                graph.run(query)  # Another common method name
                return "Database successfully flushed"
            except:
                raise e

def topic_exists(topic: str, graph) -> bool:
    query = """
    MATCH (n:Concept)
    WHERE toLower(n.name) CONTAINS toLower($topic)
    RETURN COUNT(n) > 0 AS exists
    """
    result = graph.run(query, topic=topic).evaluate()
    return result
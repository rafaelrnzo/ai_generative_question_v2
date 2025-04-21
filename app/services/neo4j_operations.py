from core.dependencies import get_graph
import logging
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.config import OLLAMA_HOST, OLLAMA_MODEL
from services.llm_services import LLMService
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from services.essay_services import EssayService
from utils.helpers import is_mcq_request

def query_rag_system(question, vector_retriever, graph):
    retrieved_docs = vector_retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    is_mcq = any(keyword in question.lower() for keyword in ['soal', 'pilihan ganda', 'mcq', 'multiple choice', 'pertanyaan'])
    
    llm_service = LLMService()
    
    try:
        if is_mcq:
            response = llm_service.generate_mcq(question, "indonesian", formatted_context)  # Added language parameter
        else:
            response = llm_service.generate_json_response(question, "indonesian", formatted_context)  # Added language parameter
        
        return {
            "status": "success",
            "query": question,
            "response": response,
            "metadata": {
                "model": llm_service.model,
                "document_chunks": len(retrieved_docs),
                "type": "mcq" if is_mcq else "general"
            }
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e)
        }
        
def query_rag_essay(question, vector_retriever, graph, language):
    retrieved_docs = vector_retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    is_essay = any(keyword in question.lower() for keyword in ['soal', 'essay', 'pertanyaan', 'question'])
    
    essay_service = EssayService()
    
    try:
        if is_essay:
            response = essay_service.generate_essay(question, language, formatted_context)
        else:
            pass
        
        return {
            "status": "success",
            "query": question,
            "response": response,
            "metadata": {
                "model": essay_service.model,
                "document_chunks": len(retrieved_docs),
                "type": "Essay" if is_essay else "General"
            }
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e)
        }
        
def query_rag_mcq(question, vector_retriever, graph, language):
    retrieved_docs = vector_retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    is_mcq = is_mcq_request(question)
    
    llm_service = LLMService()

    try:
        if is_mcq:
            response = llm_service.generate_mcq(question, language, formatted_context)
        else:
            response = llm_service.generate_json_response(question, language, formatted_context)

        return {
            "status": "success",
            "query": question,
            "response": response,
            "metadata": {
                "model": llm_service.model,
                "document_chunks": len(retrieved_docs),
                "type": "mcq" if is_mcq else "general"
            }
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e)
        }

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
    result = graph.query(query)
    return "success"

def topic_exists(topic: str, graph) -> bool:
    query = """
    MATCH (n:Concept)
    WHERE toLower(n.name) CONTAINS toLower($topic)
    RETURN COUNT(n) > 0 AS exists
    """
    result = graph.run(query, topic=topic).evaluate()
    return result

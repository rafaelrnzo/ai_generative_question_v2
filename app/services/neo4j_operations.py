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

def query_rag_system(question, vector_retriever, graph):
    retrieved_docs = vector_retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    is_mcq = any(keyword in question.lower() for keyword in ['soal', 'pilihan ganda', 'mcq', 'multiple choice'])
    
    llm_service = LLMService()
    
    try:
        if is_mcq:
            response = llm_service.generate_mcq(question, formatted_context)
        else:
            response = llm_service.generate_json_response(question, formatted_context)
        
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
        
def query_essay(question, vector_retriever, graph):
    retrieved_docs = vector_retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    is_essay = any(keyword in question.lower() for keyword in ['soal', 'essay', 'pertanyaan'])
    
    essay_service = EssayService()
    
    try:
        if is_essay:
            response = essay_service.generate_essay(question, formatted_context)
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
             
def query_rag_mcq(question, vector_retriever, graph):
    retrieved_docs = vector_retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    is_mcq = any(keyword in question.lower() for keyword in ['soal', 'pilihan ganda', 'mcq', 'multiple choice'])
    
    llm_service = LLMService()
    
    try:
        if is_mcq:
            response = llm_service.generate_mcq(question, formatted_context)
        else:
            response = llm_service.generate_json_response(question, formatted_context)
        
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
        
def delete_data_from_neo4j(name, graph):
    logging.info(f"Attempting to delete data related to: {name}")
    
    try:
        # First get a count of nodes that will be deleted
        count_query = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($name) OR 
              toLower(n.id) CONTAINS toLower($name) OR
              (n.text IS NOT NULL AND toLower(n.text) CONTAINS toLower($name))
        RETURN count(n) as count
        """
        
        count_result = graph.query(count_query, {"name": name})
        nodes_to_delete = count_result[0]["count"] if count_result else 0
        
        if nodes_to_delete == 0:
            logging.info(f"No nodes found matching: {name}")
            return 0
        
        # Log sample nodes for debugging
        check_query = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($name) OR 
              toLower(n.id) CONTAINS toLower($name) OR
              (n.text IS NOT NULL AND toLower(n.text) CONTAINS toLower($name))
        RETURN n.name, n.id, labels(n) as labels
        LIMIT 10
        """
        
        check_result = graph.query(check_query, {"name": name})
        logging.info(f"Found {len(check_result)} sample nodes matching '{name}':")
        for node in check_result:
            logging.info(f"  - {node}")
            
        # First, delete the relationships
        relationships_query = """
        MATCH (n)-[r]-(m)
        WHERE toLower(n.name) CONTAINS toLower($name) OR 
              toLower(n.id) CONTAINS toLower($name) OR
              (n.text IS NOT NULL AND toLower(n.text) CONTAINS toLower($name))
        DELETE r
        """
        
        graph.query(relationships_query, {"name": name})
        logging.info(f"Deleted relationships connected to nodes matching: {name}")
        
        # Then, delete the nodes
        delete_query = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($name) OR 
              toLower(n.id) CONTAINS toLower($name) OR
              (n.text IS NOT NULL AND toLower(n.text) CONTAINS toLower($name))
        DELETE n
        """
        
        graph.query(delete_query, {"name": name})
        
        # Verify the deletion
        verify_query = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($name) OR 
              toLower(n.id) CONTAINS toLower($name) OR
              (n.text IS NOT NULL AND toLower(n.text) CONTAINS toLower($name))
        RETURN count(n) as remaining
        """
        
        verify_result = graph.query(verify_query, {"name": name})
        remaining = verify_result[0]["remaining"] if verify_result else 0
        
        deleted_count = nodes_to_delete - remaining
        
        if remaining > 0:
            logging.warning(f"Deletion incomplete. {remaining} nodes still remain.")
        else:
            logging.info("Deletion verified. No matching nodes remain.")
        
        logging.info(f"Successfully deleted {deleted_count} nodes matching: {name}")
        return deleted_count
        
    except Exception as e:
        logging.error(f"Error deleting data for '{name}': {str(e)}")
        raise
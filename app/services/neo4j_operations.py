from core.dependencies import get_graph
import logging
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.config import OLLAMA_HOST, OLLAMA_MODEL

def delete_data_from_neo4j(filename, graph):
    result = graph.query(
        """
        MATCH (d:Document) WHERE d.source_file = $filename
        DETACH DELETE d RETURN count(d) as deleted_count
        """, {"filename": filename}
    )
    return result[0]["deleted_count"] if result else 0

def query_rag_system(question, vector_retriever, graph):
    
    retrieved_docs = vector_retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({"context": context, "question": question})
    logging.info(f"Generated answer")
    return response

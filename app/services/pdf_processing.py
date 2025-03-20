import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama
from core.config import UPLOAD_DIR, OLLAMA_HOST, OLLAMA_MODEL

logging.basicConfig(level=logging.INFO)

def load_pdf(file_path):
    logging.info(f"Loading PDF from: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    total_pages = len(documents)
    total_content = sum(len(doc.page_content) for doc in documents)
    logging.info(f"Loaded PDF with {total_pages} pages and {total_content} characters")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(split_docs)} chunks for processing")
    
    return split_docs

def store_documents(documents, graph):
    logging.info(f"Starting ingestion process for {len(documents)} documents")
    
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0)
    llm_transformer_filtered = LLMGraphTransformer(llm=llm)
    
    batch_size = 5
    processed_docs = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        try:
            graph_documents = llm_transformer_filtered.convert_to_graph_documents(batch)
            
            graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )
            processed_docs += len(batch)
            logging.info(f"Added batch {i//batch_size + 1} to the graph")
        except Exception as e:
            logging.error(f"Error processing batch {i//batch_size + 1}: {e}")
    
    logging.info(f"Successfully added {processed_docs} documents to the graph")
    return processed_docs
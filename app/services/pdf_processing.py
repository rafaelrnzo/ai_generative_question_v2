import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.config import UPLOAD_DIR

logging.basicConfig(level=logging.INFO)

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def store_documents(documents, graph):
    for doc in documents:
        graph.add_document(doc)
    return len(documents)

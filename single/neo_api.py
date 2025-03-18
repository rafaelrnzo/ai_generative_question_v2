import os
import logging
import json
import shutil
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
OLLAMA_HOST = "http://192.168.100.3:11434"
OLLAMA_MODEL = "llama3.2:latest"
UPLOAD_DIR = "uploaded_pdfs"

os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Neo4j PDF RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the RAG system")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The answer from the RAG system")

class DeleteRequest(BaseModel):
    filename: str = Field(..., description="The filename to delete from the system")
    delete_file: bool = Field(False, description="Whether to delete the PDF file itself")

class DeleteResponse(BaseModel):
    message: str = Field(..., description="Status message")
    deleted_nodes: int = Field(..., description="Number of nodes deleted")

class UploadResponse(BaseModel):
    filename: str = Field(..., description="The filename that was uploaded")
    document_count: int = Field(..., description="Number of document chunks created")
    message: str = Field(..., description="Status message")

# Database connection
def get_graph():
    return Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="admin.admin"
    )

# Vector retriever
def get_vector_retriever():
    embed = OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_HOST)
    try:
        vector_index = Neo4jVector.from_existing_graph(
            embedding=embed,
            search_type="hybrid",
            url="bolt://localhost:7687",
            username="neo4j",
            password="admin.admin",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        return vector_index.as_retriever()
    except Exception as e:
        logging.error(f"Error creating vector index: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating vector index: {str(e)}")

# Functions
def load_pdf(file_path):
    logging.info(f"Loading PDF from: {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        split_docs = text_splitter.split_documents(documents)
        logging.info(f"Split into {len(split_docs)} chunks for processing")
        
        filename = os.path.basename(file_path)
        for doc in split_docs:
            doc.metadata["source_file"] = filename
        
        return split_docs
    except Exception as e:
        logging.error(f"Error loading PDF: {e}")
        raise

def store_documents(documents, graph):
    logging.info(f"Storing {len(documents)} document chunks")
    
    for doc in documents:
        graph.add_document(
            document=doc,
            embedding_model=OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_HOST)
        )
    
    logging.info(f"Successfully added {len(documents)} document chunks to the graph")
    return len(documents)

def query_rag_system(question, vector_retriever, graph):
    logging.info(f"Querying with question: {question}")
    
    # Retrieve relevant documents
    retrieved_docs = vector_retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Generate answer with LLM
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

def delete_data_from_neo4j(filename, graph):
    logging.info(f"Deleting data related to file: {filename}")
    
    # Delete nodes and relationships related to the file
    result = graph.query(
        """
        MATCH (d:Document)
        WHERE d.source_file = $filename
        DETACH DELETE d
        RETURN count(d) as deleted_count
        """,
        {"filename": filename}
    )
    
    deleted_count = result[0]["deleted_count"] if result else 0
    logging.info(f"Deleted {deleted_count} nodes related to {filename}")
    
    return deleted_count

# API Endpoints
@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    graph: Neo4jGraph = Depends(get_graph),
    vector_retriever = Depends(get_vector_retriever)
):
    try:
        answer = query_rag_system(request.question, vector_retriever, graph)
        return QueryResponse(answer=answer)
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    graph: Neo4jGraph = Depends(get_graph)
):
    try:
        # Validate file is a PDF
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logging.info(f"File saved to {file_path}")
        
        # Process the PDF
        documents = load_pdf(file_path)
        doc_count = store_documents(documents, graph)
        
        return UploadResponse(
            filename=file.filename,
            document_count=doc_count,
            message="File uploaded and processed successfully"
        )
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete", response_model=DeleteResponse)
async def delete_data(
    request: DeleteRequest,
    graph: Neo4jGraph = Depends(get_graph)
):
    try:
        # Delete from Neo4j
        deleted_count = delete_data_from_neo4j(request.filename, graph)
        
        # Optionally delete the file
        file_path = os.path.join(UPLOAD_DIR, request.filename)
        if request.delete_file and os.path.exists(file_path):
            os.remove(file_path)
            message = f"Data deleted from Neo4j and file {request.filename} removed"
        else:
            message = f"Data deleted from Neo4j, file {request.filename} preserved"
        
        return DeleteResponse(
            message=message,
            deleted_nodes=deleted_count
        )
    except Exception as e:
        logging.error(f"Error deleting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files", response_model=List[str])
async def list_files():
    try:
        files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith('.pdf')]
        return files
    except Exception as e:
        logging.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "api_version": "1.0.0"}

@app.get("/")
async def redirect_to_docs():
    return RedirectResponse("/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
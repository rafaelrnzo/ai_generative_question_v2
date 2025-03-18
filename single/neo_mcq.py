import os
import time
import logging
import re
import json
import shutil
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from functools import lru_cache

OLLAMA_HOST = "http://192.168.100.3:11434"
OLLAMA_MODEL = "llama3.2:latest"
UPLOAD_DIR = "uploaded_pdfs"
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "admin.admin"

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

class EssayQuestion(BaseModel):
    number: int
    question: str
    answer: str
    explanation: Optional[str] = None

class MCQOption(BaseModel):
    A: str
    B: str
    C: str
    D: str

class MCQQuestion(BaseModel):
    number: int
    question: str
    options: MCQOption
    answer: str

class QuestionResponse(BaseModel):
    total_questions: int
    questions: List

@lru_cache(maxsize=1)
def get_llm():
    return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0)

@lru_cache(maxsize=1)
def get_embeddings():
    return OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_HOST)

def get_graph():
    return Neo4jGraph(
        url=NEO4J_URL,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD
    )

def get_vector_retriever():
    try:
        embed = get_embeddings()
        vector_index = Neo4jVector.from_existing_graph(
            embedding=embed,
            search_type="hybrid",
            url=NEO4J_URL,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
    except Exception as e:
        logging.error(f"Error creating vector index: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating vector index: {str(e)}")

def load_pdf(file_path):
    logging.info(f"Loading PDF from: {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        total_pages = len(documents)
        total_content = sum(len(doc.page_content) for doc in documents)
        logging.info(f"Loaded PDF with {total_pages} pages and {total_content} characters")
        
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

def ingestion(documents, graph):
    logging.info(f"Starting ingestion process for {len(documents)} documents")
    
    llm = get_llm()
    llm_transformer_filtered = LLMGraphTransformer(llm=llm)
    
    batch_size = 5
    processed_count = 0
    
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
            processed_count += len(batch)
            logging.info(f"Added batch {i//batch_size + 1} to the graph")
        except Exception as e:
            logging.error(f"Error processing batch {i//batch_size + 1}: {e}")
    
    logging.info(f"Successfully added {processed_count} document chunks to the graph")
    
    embed = get_embeddings()
    
    try:
        vector_index = Neo4jVector.from_existing_graph(
            embedding=embed,
            search_type="hybrid",
            url=NEO4J_URL,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        logging.info("Vector index created/updated successfully")
        return vector_index.as_retriever(search_kwargs={"k": 5}), processed_count
    except Exception as e:
        logging.error(f"Error creating vector index: {e}")
        raise

def extract_entities(question: str) -> List[str]:
    """Extract entities from question text using simple heuristics for speed"""
    entities = []
    words = question.split()
    for i in range(len(words)):
        word = words[i].strip('.,!?():;')
        if i == 0 and word.lower() in ["who", "what", "where", "when", "why", "how"]:
            continue
            
        if word and word[0].isupper() and word.lower() not in ["i", "the", "a", "an", "is", "are", "am"]:
            if i < len(words) - 1 and words[i+1][0].isupper():
                entities.append(f"{word} {words[i+1].strip('.,!?():;')}")
            else:
                entities.append(word)
    
    return list(set(entities))

def query_neo4j(question: str, graph: Neo4jGraph) -> str:
    entities = extract_entities(question)
    logging.info(f"Extracted entities: {entities}")
    
    if not entities:
        return "No specific entities found in the question."
    
    results = []
    for entity in entities:
        query = """
        MATCH (p)-[r]->(e)
        WHERE p.id CONTAINS $entity OR p.name CONTAINS $entity
        RETURN DISTINCT COALESCE(p.name, p.id) AS source_id, type(r) AS relationship, COALESCE(e.name, e.id) AS target_id
        LIMIT 10
        """
        
        response = graph.query(query, {"entity": entity})
        
        if response:
            results.append(f"Relationships for {entity}:")
            for rel in response:
                results.append(f"  {rel['source_id']} - {rel['relationship']} -> {rel['target_id']}")
    
    if not results:
        return "No relationships found for the entities in the question."
    
    return "\n".join(results)

def get_context(question: str, vector_retriever, graph: Neo4jGraph) -> str:
    graph_data = query_neo4j(question, graph)
    
    try:
        start = time.time()
        vector_docs = vector_retriever.invoke(question)
        end = time.time()
        logging.info(f"Vector retrieval took {end-start:.2f} seconds")
        
        vector_text = ""
        if vector_docs:
            vector_text = "\n---\n".join(doc.page_content for doc in vector_docs)
        else:
            vector_text = "No relevant documents found."
    except Exception as e:
        logging.error(f"Error in vector retrieval: {e}")
        vector_text = "Error retrieving vector data."
    
    return f"Graph data:\n{graph_data}\n\nVector data:\n{vector_text}"

def query_llm(question: str, context: str) -> str:
    template = """
    Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    
    try:
        start = time.time()
        response = chain.invoke({"context": context, "question": question})
        end = time.time()
        logging.info(f"LLM query took {end-start:.2f} seconds")
        return response
    except Exception as e:
        logging.error(f"Error in LLM query: {e}")
        return f"Error generating response: {str(e)}"

def format_mcq_prompt(question: str, context: str) -> str:
    return f"""Berdasarkan konteks berikut, buatkan soal pilihan ganda sesuai permintaan.
    
    FORMAT YANG WAJIB DIIKUTI (simpan format persis seperti ini):
    
    Soal 1:
    [Pertanyaan lengkap]
    A) [pilihan A]
    B) [pilihan B]
    C) [pilihan C]
    D) [pilihan D]
    Jawaban: [A/B/C/D]
    
    Soal 2:
    [Pertanyaan lengkap]
    A) [pilihan A]
    B) [pilihan B]
    C) [pilihan C]
    D) [pilihan D]
    Jawaban: [A/B/C/D]
    
    PENTING:
    1. HARUS menggunakan format di atas PERSIS
    2. Setiap soal harus memiliki pertanyaan lengkap
    3. Setiap opsi harus dimulai dengan A), B), C), atau D) diikuti spasi
    4. Jawaban harus dalam format "Jawaban: X" dimana X adalah A, B, C, atau D
    5. Jawaban benar harus tersebar secara merata di antara A, B, C, dan D
    
    Konteks: {context}
    
    Permintaan: {question}"""

def parse_mcq_text(content: str) -> Dict[str, Any]:
    try:
        questions = []
        parts = content.split("Soal")
        raw_questions = [part.strip() for part in parts if part.strip()]
        
        for i, raw_question in enumerate(raw_questions, 1):
            try:
                lines = [line.strip() for line in raw_question.split('\n') if line.strip()]
                
                question_text = ""
                option_lines = []
                for idx, line in enumerate(lines):
                    if line.startswith(('A)', 'A.', 'A ')):
                        option_lines = lines[idx:]
                        question_text = ' '.join(lines[:idx])
                        break
                
                question_text = re.sub(r'^\d+[:.]\s*', '', question_text).strip()
                
                options = {"A": "", "B": "", "C": "", "D": ""}
                for line in option_lines:
                    for opt in ["A", "B", "C", "D"]:
                        if line.startswith(f"{opt})") or line.startswith(f"{opt}.") or line.startswith(f"{opt} "):
                            option_text = re.sub(f"^{opt}[)\\. ]\\s*", "", line).strip()
                            options[opt] = option_text
                            break
                
                answer = None
                for line in lines:
                    if "Jawaban:" in line or "jawaban:" in line:
                        answer_match = re.search(r'[Jj]awaban:?\s*([A-D])', line)
                        if answer_match:
                            answer = answer_match.group(1)
                            break

                if question_text and all(options.values()) and answer:
                    questions.append({
                        "number": i,
                        "question": question_text,
                        "options": options,
                        "answer": answer
                    })
            except Exception as e:
                logging.error(f"Error parsing question {i}: {str(e)}")
                continue
        
        return {
            "total_questions": len(questions),
            "questions": questions
        }
    except Exception as e:
        logging.error(f"Failed to parse MCQ text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing MCQ response: {str(e)}")

def format_essay_prompt(question: str, context: str) -> str:
    """Create a prompt for essay question generation"""
    return f"""Berdasarkan konteks berikut, buatkan soal essay sesuai permintaan.
    
    FORMAT YANG WAJIB DIIKUTI:
    
    Soal 1:
    [pertanyaan lengkap]
    
    Jawaban:
    [jawaban lengkap]
    
    Penjelasan:
    [penjelasan detail]
    
    Soal 2:
    [pertanyaan lengkap]
    
    Dan seterusnya...
    
    Konteks: {context}
    
    Permintaan: {question}"""

def parse_essay_text(content: str) -> Dict[str, Any]:
    """Parse essay question text into structured JSON format"""
    try:
        questions = []
        parts = content.split("Soal")
        raw_questions = [part.strip() for part in parts if part.strip()]
        
        for i, raw_question in enumerate(raw_questions, 1):
            try:
                sections = re.split(r'\n\s*(?:Jawaban|Penjelasan):\s*', raw_question)
                
                if len(sections) >= 2:
                    question_text = re.sub(r'^\d+[:.]\s*', '', sections[0]).strip()
                    answer_text = sections[1].strip() if len(sections) > 1 else ""
                    explanation_text = sections[2].strip() if len(sections) > 2 else ""
                    
                    questions.append({
                        "number": i,
                        "question": question_text,
                        "answer": answer_text,
                        "explanation": explanation_text if explanation_text else None
                    })
            except Exception as e:
                logging.error(f"Error parsing essay question {i}: {str(e)}")
                continue
        
        return {
            "total_questions": len(questions),
            "questions": questions
        }
    except Exception as e:
        logging.error(f"Failed to parse essay text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing essay response: {str(e)}")

def generate_mcq(question: str, context: str) -> Dict[str, Any]:
    """Generate MCQ questions and ensure proper JSON formatting"""
    try:
        start = time.time()
        logging.info("Generating MCQ questions")
        llm = get_llm()
        
        prompt = format_mcq_prompt(question, context)
        response = llm.invoke(prompt)
        content = response.content
        
        parsed_result = parse_mcq_text(content)
        
        if not parsed_result["questions"]:
            logging.warning("No valid questions parsed from LLM response")
            logging.debug(f"Raw content: {content}")
            
            fixed_content = content.replace("Pilihan:", "").replace("Jawaban benar:", "Jawaban:")
            parsed_result = parse_mcq_text(fixed_content)
        
        end = time.time()
        logging.info(f"MCQ generation took {end-start:.2f} seconds, parsed {len(parsed_result['questions'])} questions")
        return parsed_result
        
    except Exception as e:
        logging.error(f"Error in generate_mcq: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate MCQ questions: {str(e)}")

def generate_essay(question: str, context: str) -> Dict[str, Any]:
    """Generate essay questions and ensure proper JSON formatting"""
    try:
        start = time.time()
        logging.info("Generating essay questions")
        llm = get_llm()
        
        prompt = format_essay_prompt(question, context)
        response = llm.invoke(prompt)
        content = response.content
        
        parsed_result = parse_essay_text(content)
        
        end = time.time()
        logging.info(f"Essay generation took {end-start:.2f} seconds, parsed {len(parsed_result['questions'])} questions")
        return parsed_result
        
    except Exception as e:
        logging.error(f"Error in generate_essay: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate essay questions: {str(e)}")

def delete_data_from_neo4j(filename: str, graph: Neo4jGraph) -> int:
    """Delete data related to a specific file from Neo4j"""
    try:
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
    except Exception as e:
        logging.error(f"Error deleting data: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting data: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    graph: Neo4jGraph = Depends(get_graph),
    vector_retriever = Depends(get_vector_retriever)
):
    try:
        context = get_context(request.question, vector_retriever, graph)
        answer = query_llm(request.question, context)
        
        return QueryResponse(answer=answer)
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    graph: Neo4jGraph = Depends(get_graph)
):
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logging.info(f"File saved to {file_path}")
        
        documents = load_pdf(file_path)
        
        background_tasks.add_task(ingestion, documents, graph)
        
        return UploadResponse(
            filename=file.filename,
            document_count=len(documents),
            message="File uploaded and processing started in the background"
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
        deleted_count = delete_data_from_neo4j(request.filename, graph)
        
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
    """List all PDF files in the upload directory"""
    try:
        files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith('.pdf')]
        return files
    except Exception as e:
        logging.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/prompt-mcq")
async def rag_chain_mcq(
    request: QueryRequest,
    graph: Neo4jGraph = Depends(get_graph),
    vector_retriever = Depends(get_vector_retriever)
):
    """Generate MCQ questions based on a query"""
    try:
        start_time = time.time()
        logging.info(f"Processing MCQ request: {request.question}")
        
        context = get_context(request.question, vector_retriever, graph)
        
        mcq_response = generate_mcq(request.question, context)
        
        end_time = time.time()
        logging.info(f"MCQ generation completed in {end_time - start_time:.2f} seconds")
        
        return JSONResponse(content={
            "status": "success",
            "query": request.question,
            "response": mcq_response,
            "metadata": {
                "model": OLLAMA_MODEL,
                "type": "mcq",
                "processing_time": f"{end_time - start_time:.2f} seconds"
            }
        })
        
    except Exception as e:
        logging.error(f"Error in MCQ endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "query": request.question
            }
        )

@app.post("/prompt-essay")
async def rag_chain_essay(
    request: QueryRequest,
    graph: Neo4jGraph = Depends(get_graph),
    vector_retriever = Depends(get_vector_retriever)
):
    try:
        start_time = time.time()
        logging.info(f"Processing essay request: {request.question}")
        
        context = get_context(request.question, vector_retriever, graph)
        
        essay_response = generate_essay(request.question, context)
        
        end_time = time.time()
        logging.info(f"Essay generation completed in {end_time - start_time:.2f} seconds")
        
        return JSONResponse(content={
            "status": "success",
            "query": request.question,
            "response": essay_response,
            "metadata": {
                "model": OLLAMA_MODEL,
                "type": "essay",
                "processing_time": f"{end_time - start_time:.2f} seconds"
            }
        })
        
    except Exception as e:
        logging.error(f"Error in essay endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "query": request.question
            }
        )

@app.post("/prompt")
async def rag_chain_legacy(
    request: QueryRequest,
    graph: Neo4jGraph = Depends(get_graph),
    vector_retriever = Depends(get_vector_retriever)
):
    return await rag_chain_mcq(request, graph, vector_retriever)

@app.post("/prompt-json")
async def rag_chain_json(
    request: QueryRequest,
    graph: Neo4jGraph = Depends(get_graph),
    vector_retriever = Depends(get_vector_retriever)
):
    try:
        is_mcq = any(keyword in request.question.lower() 
                    for keyword in ['soal', 'pilihan ganda', 'mcq', 'multiple choice'])
        
        if is_mcq:
            return await rag_chain_mcq(request, graph, vector_retriever)
        else:
            context = get_context(request.question, vector_retriever, graph)
            
            answer = query_llm(request.question, context)
            
            return JSONResponse(content={
                "status": "success",
                "query": request.question,
                "response": {
                    "answer": answer,
                    "confidence": 0.85,
                    "references": ["context-based"],
                    "tags": ["general-query"]
                },
                "metadata": {
                    "model": OLLAMA_MODEL,
                    "type": "general"
                }
            })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "query": request.question
            }
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "api_version": "1.0.0"}

@app.get("/")
async def redirect_to_docs():
    return RedirectResponse("/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
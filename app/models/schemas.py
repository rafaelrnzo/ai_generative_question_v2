from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the RAG system")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The answer from the RAG system")

class UploadResponse(BaseModel):
    filename: str
    document_count: int
    message: str

class DeleteByNameRequest(BaseModel):
    name: str
    delete_file: bool = False

class DeleteResponse(BaseModel):
    message: str
    deleted_nodes: int
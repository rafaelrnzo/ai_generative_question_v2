from pydantic import BaseModel, Field
from typing import List, Optional, Literal

Language = Literal["english", "indonesian"]

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the RAG system.")
    language: str = Field(..., description="Language to use for the query: 'english' or 'indonesian'.")

class EssayRequest(BaseModel):
    query: str = Field(..., description="The prompt for essay generation.")
    language: str = Field(default="english", description="Language for essay generation: 'english' or 'indonesian'.")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The answer from the RAG system.")

class UploadRequest(BaseModel):
    language: Language = Field(..., description="Language of the uploaded document: 'english' or 'indonesian'.")

class UploadResponse(BaseModel):
    filename: str
    language: str
    document_count: Optional[int] = None
    message: str
    
class DeleteRequest(BaseModel):
    filename: str
    language: str

class RecompiledFile(BaseModel):
    file: str
    documents_added: int

class DeleteResponse(BaseModel):
    message: str
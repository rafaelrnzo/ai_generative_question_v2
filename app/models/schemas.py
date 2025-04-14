from pydantic import BaseModel, Field
from typing import List, Optional, Literal

Language = Literal["english", "indonesian"]

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the RAG system.")
    language: str = Field(..., description="Language to use for the query: 'english' or 'indonesian'.")

class EssayRequest(BaseModel):
    question: str = Field(..., description="The prompt for essay generation.")
    language: str = Field(default="english", description="Language for essay generation: 'english' or 'indonesian'.")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The answer from the RAG system.")

class UploadRequest(BaseModel):
    language: Language = Field(..., description="Language of the uploaded document: 'english' or 'indonesian'.")

class UploadResponse(BaseModel):
    filename: str = Field(..., description="Uploaded file name.")
    language: Language = Field(..., description="Language of the uploaded document.")
    document_count: int = Field(..., description="Number of documents stored from the upload.")
    message: str = Field(..., description="Status message of the upload process.")

class DeleteByNameRequest(BaseModel):
    name: str = Field(..., description="Name of the document or file to delete.")
    delete_file: bool = Field(default=False, description="Whether to also delete the uploaded file.")

class DeleteResponse(BaseModel):
    message: str = Field(..., description="Result message of the delete operation.")
    deleted_nodes: int = Field(..., description="Number of nodes deleted from the database.")

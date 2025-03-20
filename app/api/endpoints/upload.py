from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from models.schemas import UploadResponse
from services.pdf_processing import load_pdf, store_documents
from core.config import UPLOAD_DIR
from core.dependencies import get_graph
import shutil
import os
import logging

router = APIRouter(prefix="/upload", tags=["upload"])

@router.post("/", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), graph=Depends(get_graph)):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        documents = load_pdf(file_path)
        doc_count = store_documents(documents, graph)
        
        return UploadResponse(
            filename=file.filename, 
            document_count=doc_count, 
            message="Upload successful"
        )
    except Exception as e:
        logging.error(f"Error in upload_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
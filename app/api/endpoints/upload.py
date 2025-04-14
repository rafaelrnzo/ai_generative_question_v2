from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from models.schemas import UploadResponse
from services.pdf_processing import load_pdf, store_documents
from core.config import ENGLISH_DIR, INDONESIAN_DIR  # update: import dirs
from core.dependencies import get_graph
import shutil
import os
import logging

router = APIRouter(prefix="/api/upload-file", tags=["upload"])

@router.post("/", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    language: str = Form(...)  # "english" or "indonesian"
):
    try:
        if language.lower() == "english":
            upload_dir = ENGLISH_DIR
        elif language.lower() == "indonesian":
            upload_dir = INDONESIAN_DIR
        else:
            raise HTTPException(status_code=400, detail="Unsupported language. Use 'english' or 'indonesian'.")

        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        documents = load_pdf(file_path)
        graph = get_graph(language)
        doc_count = store_documents(documents, graph)

        return UploadResponse(
            filename=file.filename,
            language=language,
            document_count=doc_count,
            message="Upload successful"
        )
    except Exception as e:
        logging.error(f"Error in upload_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

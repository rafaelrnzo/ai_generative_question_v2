from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.schemas import DeleteRequest, DeleteResponse
from services.neo4j_operations import flush_db
from utils.file_utils import process_pdf_file
from core.dependencies import get_graph_upload
from pathlib import Path
import logging
import os
from core.config import ENGLISH_DIR, INDONESIAN_DIR
from services.pdf_processing import load_pdf, store_documents

router = APIRouter(prefix="/api/delete-file", tags=["delete"])

@router.post("/", response_model=DeleteResponse)
async def delete_data(request: DeleteRequest):
    try:
        filename = request.filename.strip().lower()
        language = request.language.lower()
        graph = get_graph_upload(language)

        logging.info(f"[Delete] filename='{filename}', language='{language}'")

        folder_path = Path(ENGLISH_DIR if language == "english" else INDONESIAN_DIR)

        if not folder_path.exists():
            raise HTTPException(status_code=404, detail=f"Language folder '{language}' not found.")

        deleted_file = None
        for pdf_file in folder_path.glob("*.pdf"):
            if filename in pdf_file.stem.lower():
                os.remove(pdf_file)
                deleted_file = pdf_file.name
                logging.info(f"[Delete] Removed file: {pdf_file}")
                break

        if not deleted_file:
            raise HTTPException(status_code=404, detail=f"File matching '{filename}' not found in '{language}' folder.")

        deleted_nodes = flush_db(graph)

        recompiled_files = []
        for pdf_file in folder_path.glob("*.pdf"):
            documents = load_pdf(pdf_file)
            result = store_documents(documents, graph)
            recompiled_files.extend(result)

        return DeleteResponse(
            message=f"Deleted file '{deleted_file}' and flushed {deleted_nodes} Neo4j nodes for language '{language}'.",
        )

    except Exception as e:
        logging.exception("[Delete Error]")
        raise HTTPException(status_code=500, detail=f"Error during deletion: {str(e)}")

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

        try:
            result = flush_db(graph)
            logging.info(f"[Delete] Database flush result: {result}")
        except Exception as e:
            logging.error(f"[Delete] Error flushing database: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error flushing database: {str(e)}")

        recompiled_files = []
        remaining_files = list(folder_path.glob("*.pdf"))

        if remaining_files:
            logging.info(f"[Delete] Re-uploading {len(remaining_files)} remaining files")
            for pdf_file in remaining_files:
                try:
                    documents = load_pdf(pdf_file)
                    store_result = store_documents(documents, graph)
                    recompiled_files.append(pdf_file.name)
                    logging.info(f"[Delete] Re-uploaded file: {pdf_file}, result: {store_result}")
                except Exception as e:
                    logging.error(f"[Delete] Error re-uploading file {pdf_file}: {str(e)}")
        else:
            logging.info(f"[Delete] No files remaining to re-upload in {language} folder")

        return DeleteResponse(
            message=f"Deleted file '{deleted_file}' from '{language}' folder. "
                    f"Database flushed successfully. "
                    f"Re-uploaded {len(recompiled_files)} remaining files."
        )

    except Exception as e:
        logging.exception("[Delete Error]")
        raise HTTPException(status_code=500, detail=f"Error during deletion: {str(e)}")

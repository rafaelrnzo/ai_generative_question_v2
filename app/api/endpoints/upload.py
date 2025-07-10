from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from models.schemas import UploadResponse
from services.pdf_processing import load_pdf, store_documents_for_file
from core.config import ENGLISH_DIR, INDONESIAN_DIR
from core.dependencies import get_graph_upload
import shutil
import os
import logging
from pathlib import Path

router = APIRouter(prefix="/api/upload-file", tags=["upload"])
logger = logging.getLogger(__name__)


def process_pdf_background(processing_file_path_str: str, final_file_path_str: str, language: str, original_filename: str):
    processing_file_path = Path(processing_file_path_str)
    final_file_path = Path(final_file_path_str)

    try:
        logger.info(f"[BackgroundProcess] Task started for {original_filename} ({language}). Processing file: {processing_file_path}")
        if not processing_file_path.exists():
            logger.error(f"[BackgroundProcess] File not found: {processing_file_path}. Cannot process {original_filename}.")
            return

        documents = load_pdf(str(processing_file_path))
        if not documents:
            raise ValueError("No valid content found in the PDF.")

        graph = get_graph_upload(language)
        doc_count = store_documents_for_file(documents, graph, original_filename)

        if doc_count == 0:
            raise ValueError("PDF content could not be processed or is empty.")

        final_file_path.parent.mkdir(parents=True, exist_ok=True)
        if final_file_path.exists():
            logger.warning(f"[BackgroundProcess] Overwriting existing file: {final_file_path}")
        processing_file_path.rename(final_file_path)
        logger.info(f"[BackgroundProcess] Successfully processed and finalized '{original_filename}'.")

    except Exception as e:
        logger.error(f"[BackgroundProcess] Error during processing {original_filename} ({language}): {str(e)}", exc_info=True)
        if processing_file_path.exists():
            try:
                processing_file_path.unlink()
                logger.info(f"[BackgroundProcess] Removed invalid tmp file: {processing_file_path}")
            except Exception as unlink_err:
                logger.error(f"[BackgroundProcess] Failed to remove invalid tmp file: {unlink_err}", exc_info=True)


@router.post("/", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Form(...)
):
    original_filename = file.filename
    base_filename, ext = os.path.splitext(original_filename)

    if language.lower() == "english":
        upload_dir = Path(ENGLISH_DIR)
    elif language.lower() == "indonesian":
        upload_dir = Path(INDONESIAN_DIR)
    else:
        raise HTTPException(status_code=400, detail="Unsupported language. Use 'english' or 'indonesian'.")

    upload_dir.mkdir(parents=True, exist_ok=True)
    final_file_path = upload_dir / original_filename
    processing_file_path = upload_dir / f".{base_filename}.tmp{ext}"

    try:
        try:
            import aiofiles
            async with aiofiles.open(final_file_path, "wb") as buffer:
                content = await file.read()
                if not content:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty or unreadable.")
                await buffer.write(content)
            logger.info(f"[UploadAPI] File '{original_filename}' saved to '{final_file_path}'")
        except ImportError:
            with open(final_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"[UploadAPI] File '{original_filename}' saved (sync) to '{final_file_path}'")
        finally:
            await file.close()

        if final_file_path.exists():
            if processing_file_path.exists():
                try:
                    processing_file_path.unlink()
                except OSError as e:
                    logger.error(f"[UploadAPI] Could not delete existing processing file: {e}")
            final_file_path.rename(processing_file_path)
            logger.info(f"[UploadAPI] Renamed to '{processing_file_path.name}' for background processing.")
        else:
            raise HTTPException(status_code=500, detail="Error saving file before processing.")

        background_tasks.add_task(
            process_pdf_background,
            str(processing_file_path),
            str(final_file_path),
            language,
            original_filename
        )

        return UploadResponse(
            filename=original_filename,
            language=language,
            document_count=None,
            message=f"Upload of '{original_filename}' accepted. Processing in background (as {processing_file_path.name})."
        )

    except HTTPException as he:
        if processing_file_path.exists():
            try:
                processing_file_path.rename(final_file_path)
            except Exception as e:
                logger.error(f"[UploadAPI] Revert failed: {e}")
        raise he

    except Exception as e:
        logger.error(f"[UploadAPI] Unexpected error for '{original_filename}': {str(e)}", exc_info=True)
        if processing_file_path.exists():
            try:
                processing_file_path.rename(final_file_path)
            except Exception as e:
                logger.error(f"[UploadAPI] Revert on error failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error initiating PDF processing: {str(e)}")

from fastapi import APIRouter, HTTPException, BackgroundTasks
from models.schemas import DeleteRequest, DeleteResponse
from services.neo4j_operations import flush_db
from core.dependencies import get_graph_upload
from core.config import ENGLISH_DIR, INDONESIAN_DIR
from services.pdf_processing import load_pdf, store_documents_for_file
from pathlib import Path
import logging
import os

router = APIRouter(prefix="/api/delete-file", tags=["delete"])
logger = logging.getLogger(__name__)

def _rebuild_graph_for_language_background(language: str, folder_path_str: str, deleted_filename: str):
    try:
        logger.info(f"[RebuildTask] Starting rebuild for '{language}' due to deletion of '{deleted_filename}'")
        graph = get_graph_upload(language)
        folder_path = Path(folder_path_str)

        if not folder_path.exists() or not folder_path.is_dir():
            logger.error(f"[RebuildTask] Folder '{folder_path_str}' not found. Aborting rebuild.")
            return

        remaining_files = [f for f in folder_path.glob("*.pdf") if not f.name.startswith(".")]

        if not remaining_files:
            logger.info(f"[RebuildTask] No remaining PDF files in '{folder_path_str}' to process.")
            return

        logger.info(f"[RebuildTask] Re-processing {len(remaining_files)} files for '{language}' after flushing graph.")

        for idx, original_file_path in enumerate(remaining_files, start=1):
            original_filename = original_file_path.name
            base_filename, ext = os.path.splitext(original_filename)
            processing_file_path = original_file_path.parent / f".{base_filename}.tmp{ext}"
            renamed_for_processing = False

            try:
                logger.info(f"[RebuildTask] File {idx}/{len(remaining_files)}: Starting processing for '{original_filename}'")

                if processing_file_path.exists():
                    try:
                        processing_file_path.unlink(missing_ok=True)
                    except OSError as e_del_tmp:
                        logger.error(f"[RebuildTask] Could not remove stale temp file '{processing_file_path.name}': {e_del_tmp}.")
                        continue

                if original_file_path.exists():
                    try:
                        original_file_path.replace(processing_file_path)
                        renamed_for_processing = True
                        logger.info(f"[RebuildTask] Renamed '{original_filename}' to '{processing_file_path.name}' for processing.")
                    except Exception as e_rename_to_tmp:
                        logger.error(f"[RebuildTask] Failed to rename '{original_filename}': {e_rename_to_tmp}.", exc_info=True)
                        continue
                elif processing_file_path.exists():
                    logger.info(f"[RebuildTask] Found existing processing file '{processing_file_path.name}' for '{original_filename}'.")
                    renamed_for_processing = True
                else:
                    logger.warning(f"[RebuildTask] File '{original_filename}' not found. Skipping.")
                    continue

                documents = load_pdf(str(processing_file_path))
                doc_count = store_documents_for_file(documents, graph, original_filename)
                logger.info(f"[RebuildTask] Processed '{original_filename}' - {doc_count} documents added.")

            except FileNotFoundError:
                logger.warning(f"[RebuildTask] File not found during processing of '{original_filename}'.", exc_info=True)
            except Exception as e_process:
                logger.error(f"[RebuildTask] Error processing '{original_filename}': {e_process}", exc_info=True)
            finally:
                if renamed_for_processing and processing_file_path.exists():
                    try:
                        processing_file_path.replace(original_file_path)
                        logger.info(f"[RebuildTask] Renamed '{processing_file_path.name}' back to '{original_filename}'.")
                    except Exception as e_rename_back:
                        logger.error(f"[RebuildTask] Failed to rename back '{processing_file_path.name}': {e_rename_back}.", exc_info=True)
                elif renamed_for_processing and not processing_file_path.exists():
                    logger.warning(f"[RebuildTask] Processing file '{processing_file_path.name}' not found.")

        logger.info(f"[RebuildTask] Rebuild task finished for '{language}'.")

    except Exception as e_fatal:
        logger.error(f"[RebuildTask] Fatal error during rebuild for '{language}': {e_fatal}", exc_info=True)

@router.post("/", response_model=DeleteResponse)
async def delete_data(request: DeleteRequest, background_tasks: BackgroundTasks):
    try:
        filename = request.filename.strip().lower()
        language = request.language.lower()
        logger.info(f"[DeleteAPI] Request: filename='{filename}', language='{language}'")

        folder_map = {"english": ENGLISH_DIR, "indonesian": INDONESIAN_DIR}
        target_folder_path_str = folder_map.get(language)

        if not target_folder_path_str:
            raise HTTPException(status_code=400, detail=f"Language '{language}' not supported.")

        target_path = Path(target_folder_path_str)
        if not target_path.exists() or not target_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Target folder for '{language}' not found.")

        deleted_file_actual_name = None
        file_to_delete_path = None

        for file_in_dir in target_path.glob("*.pdf"):
            if file_in_dir.name.startswith("."):
                continue
            if file_in_dir.name.lower() == filename:
                file_to_delete_path = file_in_dir
                deleted_file_actual_name = file_in_dir.name
                break

        if not file_to_delete_path:
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

        try:
            file_to_delete_path.unlink()
            logger.info(f"[DeleteAPI] Deleted file: '{deleted_file_actual_name}'")
        except Exception as e_delete:
            logger.error(f"[DeleteAPI] Failed to delete '{deleted_file_actual_name}': {e_delete}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error deleting file '{deleted_file_actual_name}'.")

        try:
            graph = get_graph_upload(language)
            flush_db(graph)
            logger.info(f"[DeleteAPI] Graph flushed for '{language}'.")
        except Exception as e_flush:
            logger.error(f"[DeleteAPI] Graph flush failed: {e_flush}", exc_info=True)
            raise HTTPException(status_code=500, detail="File deleted, but graph flush failed.")

        background_tasks.add_task(
            _rebuild_graph_for_language_background,
            language,
            str(target_path),
            deleted_file_actual_name
        )
        logger.info(f"[DeleteAPI] Background rebuild task scheduled for '{language}'.")

        return DeleteResponse(
            message=(
                f"Successfully deleted file '{deleted_file_actual_name}' for language '{language}'. "
                f"Database flushed. Background reprocessing started."
            )
        )

    except HTTPException:
        raise
    except Exception as e_unexpected:
        logger.error(f"[DeleteAPI] Unexpected error: {e_unexpected}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error during deletion process.")

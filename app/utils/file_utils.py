from pathlib import Path
from services.pdf_processing import load_pdf, store_documents_for_file # or store_documents
from core.dependencies import get_graph_upload
import logging

logger_utils = logging.getLogger(__name__ + ".utils")

def recompile_folder_synchronously(language: str, folder_path_str: str):
    logger_utils.info(f"Starting synchronous recompilation for language '{language}', folder '{folder_path_str}'")
    recompiled_files_details = []
    graph = get_graph_upload(language)
    folder = Path(folder_path_str)

    if not folder.exists() or not folder.is_dir():
        logger_utils.warning(f"Folder not found or not a directory: {folder_path_str}")
        return {"status": "error", "message": "Folder not found", "recompiled_files": []}

    for pdf_file in folder.glob("*.pdf"):
        try:
            logger_utils.info(f"Synchronously recompiling {pdf_file.name}")
            documents = load_pdf(str(pdf_file))
            result_count = store_documents_for_file(documents, graph, pdf_file.name) 
            recompiled_files_details.append({"filename": pdf_file.name, "documents_added": result_count}) # Adapt as per return
            logger_utils.info(f"Successfully recompiled {pdf_file.name}, result: {result_count}")
        except Exception as e:
            logger_utils.error(f"Failed to recompile {pdf_file.name} synchronously: {e}", exc_info=True)
            recompiled_files_details.append({"filename": pdf_file.name, "error": str(e)})
    
    logger_utils.info(f"Synchronous recompilation finished for {language} in {folder_path_str}.")
    return {"status": "completed", "recompiled_files_details": recompiled_files_details}

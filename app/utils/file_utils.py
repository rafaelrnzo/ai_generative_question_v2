from pathlib import Path
from services.pdf_processing import load_pdf, store_documents
from core.dependencies import get_graph_upload
import logging

def recompile_folder(language: str, folder_path: str):
    logging.info(f"Recompiling all PDFs in folder: {folder_path}")
    recompiled_files = []

    graph = get_graph_upload(language)
    folder = Path(folder_path)

    if not folder.exists():
        logging.warning(f"Folder not found: {folder_path}")
        return []

    for file in folder.glob("*.pdf"):
        try:
            logging.info(f"Recompiling {file.name}")
            documents = load_pdf(str(file))
            store_documents(documents, graph)
            recompiled_files.append(file.name)
        except Exception as e:
            logging.error(f"Failed to recompile {file.name}: {e}")

    return recompiled_files

def process_pdf_file(file_path: Path, graph):
    from services.pdf_processing import load_pdf, store_documents
    documents = load_pdf(str(file_path))
    store_documents(documents, graph)

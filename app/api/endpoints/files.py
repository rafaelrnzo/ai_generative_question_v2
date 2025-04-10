from fastapi import APIRouter
import os
from core.config import UPLOAD_DIR

router = APIRouter(prefix="/api/files", tags=["files"])

@router.get("/")
async def list_files():
    response = []
    for root, dirs, files in os.walk(UPLOAD_DIR):
        for file in files:
            if file.endswith('.pdf'):
                relative_path = os.path.relpath(root, UPLOAD_DIR)
                language = relative_path.split(os.sep)[0] if relative_path != "." else "unknown"

                response.append({
                    "title": os.path.splitext(file)[0],
                    "url_file": f"/{relative_path}/{file}".replace("\\", "/"),
                    "language": language
                })

    return {
        "status": "success",
        "response": [
            {"total_files": len(response)},
            {"files": response}
        ]
    }

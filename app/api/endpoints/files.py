from fastapi import APIRouter, Query
import os
from core.config import UPLOAD_DIR
from typing import Optional

router = APIRouter(prefix="/api/files", tags=["files"])

@router.get("/")
async def list_files(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),
    search: Optional[str] = Query(None)
):
    response = []

    for root, dirs, files in os.walk(UPLOAD_DIR):
        for file in files:
            if file.endswith('.pdf'):
                if search and search.lower() not in file.lower():
                    continue

                relative_path = os.path.relpath(root, UPLOAD_DIR)
                language = relative_path.split(os.sep)[0] if relative_path != "." else "unknown"

                response.append({
                    "title": os.path.splitext(file)[0],
                    "url_file": f"/{relative_path}/{file}".replace("\\", "/"),
                    "language": language
                })

    total_files = len(response)

    start = (page - 1) * limit
    end = start + limit
    paginated_files = response[start:end]

    return {
        "status": "success",
        "response": {
            "page": page,
            "limit": limit,
            "total_files": total_files,
            "files": paginated_files
        }
    }

from fastapi import APIRouter, Query
import os
import math
import datetime
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

                file_path = os.path.join(root, file)
                created_timestamp = os.path.getctime(file_path)
                created_at = datetime.datetime.fromtimestamp(created_timestamp).isoformat()

                response.append({
                    "title": os.path.splitext(file)[0],
                    "url_file": f"/{relative_path}/{file}".replace("\\", "/"),
                    "language": language,
                    "created_at": created_at
                })

    total_files = len(response)
    total_pages = math.ceil(total_files / limit) if total_files > 0 else 1

    start = (page - 1) * limit
    end = start + limit
    paginated_files = response[start:end]

    return {
        "status": "success",
        "response": {
            "page": page,
            "limit": limit,
            "total_files": total_files,
            "total_pages": total_pages,
            "files": paginated_files
        }
    }
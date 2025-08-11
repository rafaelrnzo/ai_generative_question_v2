from fastapi import APIRouter, Query, Request
from typing import Optional
import os
import math
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from core.config import UPLOAD_DIR

router = APIRouter(prefix="/api/files", tags=["files"])

@router.get("/")
async def list_files(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),
    search: Optional[str] = Query(None),
    language: Optional[str] = Query(None, description="Filter by language (e.g., english or indonesian)"),
    status: Optional[str] = Query(None, description="Filter by status: processing or complete")
):
    response = []

    for root, _, files in os.walk(UPLOAD_DIR):
        for file in files:
            if not file.endswith('.pdf'):
                continue

            file_path = Path(root) / file
            relative_path = file_path.relative_to(UPLOAD_DIR)
            lang = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"

            if language and lang.lower() != language.lower():
                continue

            filename = file_path.stem
            is_processing = filename.endswith(".tmp") 
            clean_title = filename[:-4] if is_processing else filename

            if search and search.lower() not in clean_title.lower():
                continue

            file_status = "processing" if is_processing else "complete"
            if status and status.lower() != file_status:
                continue

            created_timestamp = file_path.stat().st_ctime
            created_at = datetime.datetime.fromtimestamp(
                created_timestamp, tz=ZoneInfo("Asia/Jakarta")
            ).isoformat()

            file_url = f"/uploads/{relative_path.as_posix()}"
            preview_url = f"{request.base_url}{file_url.lstrip('/')}"

            response.append({
                "title": clean_title,
                "url_file": file_url,
                "preview_url": preview_url,
                "language": lang,
                "status": file_status,
                "created_at": created_at
            })

    total_files = len(response)
    total_pages = max(math.ceil(total_files / limit), 1)
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

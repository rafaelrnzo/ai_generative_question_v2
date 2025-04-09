from fastapi import APIRouter
import os
from core.config import UPLOAD_DIR

router = APIRouter(prefix="/api/files", tags=["files"])

@router.get("/")
async def list_files():
    return [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.pdf')]
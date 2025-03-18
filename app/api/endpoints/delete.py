from fastapi import APIRouter, Depends
from models.schemas import DeleteRequest, DeleteResponse
from services.neo4j_operations import delete_data_from_neo4j
from core.config import UPLOAD_DIR
from core.dependencies import get_graph

import os

router = APIRouter(prefix="/delete", tags=["delete"])

@router.post("/", response_model=DeleteResponse)
async def delete_data(request: DeleteRequest, graph=Depends(get_graph)):
    deleted_count = delete_data_from_neo4j(request.filename, graph)
    file_path = os.path.join(UPLOAD_DIR, request.filename)
    if request.delete_file and os.path.exists(file_path):
        os.remove(file_path)
    return DeleteResponse(message=f"Deleted {deleted_count} nodes", deleted_nodes=deleted_count)

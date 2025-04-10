from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from models.schemas import DeleteResponse
from services.neo4j_operations import delete_data_from_neo4j  # Import the function
from core.dependencies import get_graph
import logging

class DeleteRequest(BaseModel):
    filename: str
    delete_file: bool = False

router = APIRouter(prefix="/api/delete-file", tags=["delete"])

@router.post("/", response_model=DeleteResponse)
async def delete_data(request: DeleteRequest, graph=Depends(get_graph)):
    try:
        logging.info(f"Delete request received for: {request.filename}")
        
        deleted_count = delete_data_from_neo4j(request.filename, graph)
        
        if deleted_count == 0:
            logging.warning(f"No nodes found for deletion with keyword: {request.filename}")
            return DeleteResponse(
                message=f"No nodes found matching '{request.filename}'", 
                deleted_nodes=0
            )
        
        return DeleteResponse(
            message=f"Successfully deleted {deleted_count} nodes related to '{request.filename}'", 
            deleted_nodes=deleted_count
        )
    except Exception as e:
        logging.error(f"Error during deletion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during deletion: {str(e)}")
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from models.schemas import QueryRequest, EssayRequest
from core.dependencies import get_graph, get_vector_retriever, get_vector_retriever_en
# from services.neo4j_operations import query_rag_system, query_rag_essay, query_rag_mcq
from utils.helpers import is_mcq_request
from services.essay_services import EssayService
from services.mcq_services import MCQService
import re 
import traceback

router = APIRouter(prefix="/api", tags=["query"])

@router.post("/query-essay")
def generate_essay(request: EssayRequest):
    service = EssayService(request.language)
    result = service.run(request.query)
    return result

@router.post("/query-mcq")
def generate_mcq(request: EssayRequest):
    service = MCQService(request.language)
    result = service.run(request.query)
    return result


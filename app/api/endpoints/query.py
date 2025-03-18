from fastapi import APIRouter, Depends
from models.schemas import QueryRequest, QueryResponse
from core.dependencies import get_graph, get_vector_retriever
from services.neo4j_operations import query_rag_system

router = APIRouter(prefix="/query", tags=["query"])

@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest, graph=Depends(get_graph), vector_retriever=Depends(get_vector_retriever)):
    answer = query_rag_system(request.question, vector_retriever, graph)
    return QueryResponse(answer=answer)

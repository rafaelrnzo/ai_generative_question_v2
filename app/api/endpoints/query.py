from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from models.schemas import QueryRequest
from core.dependencies import get_graph, get_vector_retriever
from services.neo4j_operations import query_rag_system
from services.llm_services import LLMService
import re 

router = APIRouter(prefix="", tags=["query"])

@router.post("/query-json")
async def query_json(request: QueryRequest, graph=Depends(get_graph), vector_retriever=Depends(get_vector_retriever)):
    try:
        collection_name = getattr(request, 'collection_name', None)
        
        if hasattr(request, 'collection_name') and collection_name:
            docs = vector_retriever.retrieve_docs(request.question, collection_name)
            
            # Check if no relevant documents were found
            if not docs:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": "No relevant information found for this question."
                    }
                )
                
            formatted_context = vector_retriever.combine_docs(docs)
            
            llm_service = LLMService()
            
            is_mcq = any(keyword in request.question.lower() for keyword in ['soal', 'pilihan ganda', 'mcq', 'multiple choice'])
            
            if is_mcq:
                print(f"Processing MCQ request: {request.question}")
                json_response = llm_service.generate_mcq(request.question, formatted_context)
                
                if json_response["total_questions"] == 0:
                    print("No questions were parsed successfully")
                    return JSONResponse(
                        status_code=400, 
                        content={
                            "status": "error", 
                            "message": "Failed to generate valid MCQ questions. Please try again with a clearer prompt."
                        }
                    )
                
                print(f"Successfully generated {json_response['total_questions']} MCQ questions")
                
                expected = 10  
                num_match = re.search(r'(\d+)\s*(?:soal|pertanyaan|question)', request.question, re.IGNORECASE)
                if num_match:
                    expected = int(num_match.group(1))
                
                if json_response["total_questions"] < expected:
                    print(f"Warning: Generated {json_response['total_questions']} questions, but {expected} were requested")
                    json_response["warning"] = f"Hanya {json_response['total_questions']} soal yang berhasil dibuat dari {expected} yang diminta."
            else:
                json_response = llm_service.generate_json_response(request.question, formatted_context)
                
                # Check if response is None or null
                if json_response is None:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "status": "error",
                            "message": "Question is out of context or cannot be answered with available information."
                        }
                    )
            
            if json_response is None or (isinstance(json_response, dict) and not json_response):
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": "Failed to generate a valid response."
                    }
                )
                
            return JSONResponse(content={
                "status": "success",
                "query": request.question,
                "collection_name": collection_name,
                "response": json_response,
                "metadata": {
                    "model": llm_service.model,
                    "document_chunks": len(docs),
                    "type": "mcq" if is_mcq else "general"
                }
            })
        else:
            result = query_rag_system(request.question, vector_retriever, graph)
            
            if result is None or (isinstance(result, dict) and not result) or result.get("response") is None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": "Question is out of context or cannot be answered with available information."
                    }
                )
                
            return JSONResponse(content=result)

    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"Error in query_json: {str(e)}")
        print(f"Traceback: {trace}")
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": str(e), "trace": trace if trace else None}
        )
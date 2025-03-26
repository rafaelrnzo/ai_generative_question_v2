from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from models.schemas import QueryRequest, EssayRequest
from core.dependencies import get_graph, get_vector_retriever
from services.neo4j_operations import query_rag_system, query_essay
from services.llm_services import LLMService
from services.essay_services import EssayService
import re 
import traceback

router = APIRouter(prefix="", tags=["query"])

# @router.post("/query-essay")
# async def query_essay(request: EssayRequest, graph=Depends(get_graph), vector_retriever=Depends(get_vector_retriever)):
#     try:
#         # Default to 1 essay if not specified
#         num_essays = 1
#         num_match = re.search(r'(\d+)\s*(?:soal|pertanyaan|essay|question)', request.question, re.IGNORECASE)
#         if num_match:
#             num_essays = int(num_match.group(1))
#             print(f"Detected request for {num_essays} essays")

#         language = getattr(request, 'language', "english")
#         collection_name = getattr(request, 'collection_name', None)

#         # Retrieve relevant documents
#         docs = vector_retriever.retrieve_docs(request.question, collection_name) if collection_name else []

#         if not docs:
#             print("No relevant documents found, falling back to graph query.")
#             result = query_rag_system(request.question, vector_retriever, graph)
#             if not result or not result.get("response"):
#                 return JSONResponse(
#                     status_code=400,
#                     content={"status": "error", "message": "No relevant information found for this question."}
#                 )
#             formatted_context = result["response"]
#         else:
#             formatted_context = vector_retriever.combine_docs(docs)

#         # Generate essays
#         essay_service = EssayService()
#         essay_response = essay_service.generate_essays(
#             request.question,
#             formatted_context,
#             num_essays,
#             language
#         )

#         # Filter out irrelevant questions using vector retrieval
#         relevant_questions = [
#             q for q in essay_response.get("questions", [])
#             if vector_retriever.retrieve_docs(q["question"], collection_name)  # Check if the question has context
#         ]

#         if not relevant_questions:
#             return JSONResponse(
#                 status_code=400,
#                 content={"status": "error", "message": "Failed to generate relevant essay questions."}
#             )

#         essay_response["questions"] = relevant_questions
#         essay_response["total_questions"] = len(relevant_questions)

#         return JSONResponse(content={
#             "status": "success",
#             "query": request.question,
#             "language": language,
#             "response": essay_response,
#             "metadata": {
#                 "model": essay_service.model,
#                 "document_chunks": len(docs),
#                 "type": "essay"
#             }
#         })
    
#     except Exception as e:
#         import traceback
#         trace = traceback.format_exc()
#         print(f"Error in query_essay: {str(e)}")
#         print(f"Traceback: {trace}")
#         return JSONResponse(
#             status_code=500, 
#             content={"status": "error", "message": str(e), "trace": trace if trace else None}
#         )

@router.post("/query-json")
async def query_json(request: QueryRequest, graph=Depends(get_graph), vector_retriever=Depends(get_vector_retriever)):
    try:
        collection_name = getattr(request, 'collection_name', None)
        
        if hasattr(request, 'collection_name') and collection_name:
            docs = vector_retriever.retrieve_docs(request.question, collection_name)
            
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
        
@router.post("/query-essay-v2")
async def query_essay_v2(
    request: QueryRequest, 
    essay_service: EssayService = Depends(), 
    graph=Depends(get_graph), 
    vector_retriever=Depends(get_vector_retriever)
):
    try:
        collection_name = getattr(request, 'collection_name', None)

        if collection_name:
            docs = vector_retriever.retrieve_docs(request.question, collection_name)

            if not docs:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "No relevant information found for this question."}
                )

            formatted_context = vector_retriever.combine_docs(docs)

            is_essay = any(keyword in request.question.lower() for keyword in ['soal', 'pertanyaan', 'essay'])

            if is_essay:
                print(f"Processing Essay request: {request.question}")
                json_response = essay_service.generate_essay(request.question, formatted_context)

                if json_response.get("total_questions", 0) == 0:
                    print("No questions were parsed successfully.")
                    return JSONResponse(
                        status_code=400,
                        content={"status": "error", "message": "Failed to generate valid essay question. Please try again with a clearer prompt."}
                    )

                print(f"Successfully generated {json_response['total_questions']} essay questions.")

                expected = 10
                num_match = re.search(r'(\d+)\s*(?:soal|pertanyaan|question)', request.question, re.IGNORECASE)
                if num_match:
                    expected = int(num_match.group(1))

                # Filtering out MCQs (if "A)" exists, it's likely an MCQ)
                filtered_questions = [
                    q for q in json_response["questions"] if not re.search(r"\bA\)", q, re.IGNORECASE)
                ]

                json_response["questions"] = filtered_questions
                json_response["total_questions"] = len(filtered_questions)

                if json_response["total_questions"] < expected:
                    print(f"Warning: Generated {json_response['total_questions']} essay questions, but {expected} were requested.")
                    json_response["warning"] = f"Hanya {json_response['total_questions']} soal yang berhasil dibuat dari {expected} yang diminta."

                return JSONResponse(content=json_response)

        # Default fallback if collection_name is missing
        result = query_essay(request.question, vector_retriever, graph)

        if not result or not isinstance(result, dict) or result.get("response") is None:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Question is out of context."}
            )

        return JSONResponse(content=result)

    except Exception as e:
        trace = traceback.format_exc()
        print(f"Error in query_essay_v2: {str(e)}\nTraceback: {trace}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "trace": trace}
        )
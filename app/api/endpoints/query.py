from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from models.schemas import QueryRequest, EssayRequest
from core.dependencies import get_graph, get_vector_retriever, get_vector_retriever_en
from services.neo4j_operations import query_rag_system, query_rag_essay, query_rag_mcq
from services.llm_services import LLMService
from utils.helpers import is_mcq_request
from services.essay_services import EssayService
import re 
import traceback

router = APIRouter(prefix="/api", tags=["query"])

@router.post("/query-essay")
async def query_essay(request: QueryRequest, graph=Depends(get_graph)):
    try:
        question = request.question
        language = request.language.lower() if request.language else 'indonesian'
        collection = getattr(request, 'collection_name', None)

        vector_retriever = get_vector_retriever_en() if language == 'english' else get_vector_retriever()

        is_essay = any(k in question.lower() for k in ['essay', 'soal', 'pertanyaan'])

        if collection:
            docs = vector_retriever.retrieve_docs(question, collection)
            if not docs:
                return JSONResponse(status_code=400, content={"status": "error", "message": "No relevant information found."})

            context = vector_retriever.combine_docs(docs)
            essay_service = EssayService()

            if is_essay:
                print(f"Processing Essay request: {question}")
                response = essay_service.generate_essay(question, context)

                if response.get("total_questions", 0) == 0:
                    return JSONResponse(status_code=400, content={"status": "error", "message": "Failed to generate valid essay questions."})

                filtered = [q for q in response["questions"] if not re.search(r"\bA\)", q, re.IGNORECASE)]
                response["questions"] = filtered
                response["total_questions"] = len(filtered)

                expected = int(re.search(r'(\d+)\s*(?:soal|pertanyaan|question)', question, re.I).group(1)) if re.search(r'(\d+)\s*(?:soal|pertanyaan|question)', question, re.I) else 10
                if response["total_questions"] < expected:
                    response["warning"] = f"Hanya {response['total_questions']} soal yang berhasil dibuat dari {expected} yang diminta."

                return JSONResponse(content={
                    "status": "success",
                    "query": question,
                    "collection_name": collection,
                    "response": response,
                    "metadata": {
                        "model": "EssayService",
                        "document_chunks": len(docs),
                        "type": "essay"
                    }
                })

        result = query_rag_essay(question, vector_retriever, graph, language)

        if not result or not result.get("response"):
            return JSONResponse(status_code=400, content={"status": "error", "message": "Question is out of context or unanswerable."})

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "trace": traceback.format_exc()}
        )

@router.post("/query-mcq")
async def query_json(request: QueryRequest, graph=Depends(get_graph)):
    try:
        question = request.question
        language = request.language.lower() if request.language else 'indonesian'
        collection = getattr(request, 'collection_name', None)

        vector_retriever = get_vector_retriever_en() if language == 'english' else get_vector_retriever()

        is_mcq = is_mcq_request(question)

        if collection:
            docs = vector_retriever.retrieve_docs(question, collection)
            if not docs:
                return JSONResponse(status_code=400, content={"status": "error", "message": "No relevant information found."})

            context = vector_retriever.combine_docs(docs)
            llm = LLMService()

            if is_mcq:
                print(f"Processing MCQ request: {question}")
                response = llm.generate_mcq(question, language, context)

                if response["total_questions"] == 0:
                    return JSONResponse(status_code=400, content={"status": "error", "message": "Failed to generate valid MCQs."})

                expected = int(re.search(r'(\d+)\s*(?:soal|pertanyaan|question)', question, re.I).group(1)) if re.search(r'(\d+)\s*(?:soal|pertanyaan|question)', question, re.I) else 10
                if response["total_questions"] < expected:
                    response["warning"] = f"Hanya {response['total_questions']} soal yang berhasil dibuat dari {expected} yang diminta."

            else:
                response = llm.generate_json_response(question, language, context)
                if not response:
                    return JSONResponse(status_code=400, content={"status": "error", "message": "Question is out of context or unanswerable."})

            return JSONResponse(content={
                "status": "success",
                "query": question,
                "collection_name": collection,
                "response": response,
                "metadata": {
                    "model": llm.model,
                    "document_chunks": len(docs),
                    "type": "mcq" if is_mcq else "general"
                }
            })

        fallback_func = query_rag_mcq if is_mcq else query_rag_system
        result = fallback_func(question, vector_retriever, graph, language=language)

        if not result or not result.get("response"):
            return JSONResponse(status_code=400, content={"status": "error", "message": "Question is out of context or unanswerable."})

        return JSONResponse(content=result)

    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "trace": traceback.format_exc()
            }
        )

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from uuid import uuid4
from core.config import OLLAMA_MODEL, OLLAMA_HOST
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableMap
from typing import Optional
from core.dependencies import get_vector_retriever_en, get_vector_retriever
from fastapi import HTTPException


class Essay(BaseModel):
    question: str = Field(description="A realistic and informative essay question.")
    answer: str = Field(description="A concise answer to the question.")


class TopicRelevance(BaseModel):
    is_relevant: bool = Field(description="Whether the query is relevant to the available context")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reason: str = Field(description="Brief explanation of the relevance assessment")


class EssayService:
    def __init__(self, language: str):
        self.model_name = OLLAMA_MODEL
        self.language = language

        self.model = ChatOllama(
            base_url=OLLAMA_HOST,
            model=OLLAMA_MODEL,
            options={
                "num_ctx": 1024,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 40,
                "cache": False,
                "seed": -1
            }
        )

        self.parser = JsonOutputParser(pydantic_object=Essay)
        self.topic_parser = JsonOutputParser(pydantic_object=TopicRelevance)

        topic_check_template_en = (
            "You are a topic relevance checker for an essay question generator.\n"
            "Consider the query relevant if:\n"
            "1. It asks to generate/create/make essay questions (always relevant)\n"
            "2. It's about a topic that can be found in the provided context\n"
            "3. It's a general request for essays/questions without specific topic (always relevant)\n\n"
            "Context:\n{context}\n\n"
            "User query: {query}\n\n"
            "Respond ONLY in this exact JSON format:\n"
            "{{\n"
            '    "is_relevant": true/false,\n'
            '    "confidence": 0.0-1.0,\n'
            '    "reason": "Brief explanation"\n'
            "}}"
        )

        topic_check_template_id = (
            "Kamu adalah pemeriksa relevansi topik untuk generator pertanyaan esai.\n"
            "Anggap pertanyaan relevan jika:\n"
            "1. Meminta untuk membuat/menghasilkan pertanyaan esai (selalu relevan)\n"
            "2. Tentang topik yang bisa ditemukan dalam konteks yang diberikan\n"
            "3. Permintaan umum untuk esai/pertanyaan tanpa topik spesifik (selalu relevan)\n\n"
            "Konteks:\n{context}\n\n"
            "Pertanyaan pengguna: {query}\n\n"
            "Balas HANYA dalam format JSON berikut:\n"
            "{{\n"
            '    "is_relevant": true/false,\n'
            '    "confidence": 0.0-1.0,\n'
            '    "reason": "Penjelasan singkat"\n'
            "}}"
        )

        self.topic_prompt = PromptTemplate(
            template=topic_check_template_en if language == "english" else topic_check_template_id,
            input_variables=["query", "context"]
        )

        prompt_template_en = (
            "You are an API that responds ONLY with a valid JSON object, generating ONE high-quality, clear, and concise essay question and its answer.\n"
            "The question must be SPECIFIC, STANDALONE, and show depth and relevance to the provided context. Avoid overly general or vague questions.\n"
            "The answer must be well-structured, informative, and demonstrate deep understanding of the context.\n"
            "DO NOT add any explanations or extra text outside the requested JSON format.\n\n"
            "Use the following context to generate the question and answer:\n"
            "Context:\n{context}\n\n"
            "User query: {query}\n\n"
            "Return exactly in the following JSON format:\n"
            '{{\n'
            '  "question": "A clear, specific, and relevant essay question based on the context.",\n'
            '  "answer": "A well-structured, informative answer that demonstrates deep understanding of the question."\n'
            '}}'
        )

        prompt_template_id = (
            "Kamu adalah API yang hanya membalas dengan JSON, menghasilkan SATU pertanyaan esai berkualitas tinggi dan jawabannya.\n"
            "Buat pertanyaan yang SINGKAT, JELAS, dan PADAT, namun tetap memiliki KEDALAMAN dan RELEVANSI dengan konteks.\n"
            "Pastikan pertanyaan tidak terlalu umum, dan jawabannya harus terstruktur, informatif, serta menunjukkan pemahaman mendalam terhadap konteks.\n"
            "JANGAN tambahkan penjelasan atau teks lain di luar fomrat JSON yang diminta.\n\n"
            "Gunakan konteks berikut untuk membuat pertanyaan dan jawaban:\n"
            "Konteks:\n{context}\n\n"
            "Permintaan pengguna: {query}\n\n"
            "Kembalikan dalam format JSON persis seperti ini:\n"
            '{{\n'
            '  "question": "Pertanyaan esai yang singkat, jelas, padat, namun mendalam dan relevan dengan konteks.",\n'
            '  "answer": "Jawaban yang terstruktur, informatif, dan menunjukkan pemahaman mendalam terhadap pertanyaan."\n'
            '}}'
        )

        self.prompt = PromptTemplate(
            template=prompt_template_en if language == "english" else prompt_template_id,
            input_variables=["query", "context"]
        )

        self.retriever = get_vector_retriever_en() if language == "english" else get_vector_retriever()

        self.topic_chain = RunnableMap({
            "context": lambda x: self.retriever.get_relevant_documents(x["query"]),
            "query": lambda x: x["query"]
        }) | self.topic_prompt | self.model | self.topic_parser

        self.essay_chain = RunnableMap({
            "context": lambda x: self.retriever.get_relevant_documents(x["query"]),
            "query": lambda x: x["query"]
        }) | self.prompt | self.model | self.parser

    def _is_essay_generation_request(self, query: str) -> bool:
        essay_keywords = [
            "make", "create", "generate", "give me", "show me", "write", "essay", "essays", 
            "question", "questions", "essay question", "essay questions", "one essay", 
            "some essays", "composition", "paper", "writing", "discuss", "explain",
            "buat", "buatkan", "bikin", "bikinkan", "kasih", "berikan", "tulis", "tuliskan",
            "esai", "essay", "soal", "pertanyaan", "pertanyaan esai", "soal esai", 
            "satu esai", "karangan", "tulisan", "bahas", "jelaskan"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in essay_keywords)

    def _get_out_of_topic_response(self, query: str, reason: str = ""):
        if self.language == "english":
            message = f"Your query '{query}' appears to be out of topic or not related to the available knowledge base."
            if reason:
                message += f" Reason: {reason}"
        else:
            message = f"Pertanyaan '{query}' tampaknya di luar topik atau tidak terkait dengan basis pengetahuan yang tersedia."
            if reason:
                message += f" Alasan: {reason}"

        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "query": query,
                "message": message,
                "response": None,
                "metadata": {
                    "model": self.model_name
                }
            }
        )

    def run(self, query: str, relevance_threshold: float = 0.3):
        random_id = str(uuid4())[:8]
        full_query = f"({random_id}) {query}"

        try:
            if self._is_essay_generation_request(query):
                print("Detected as essay generation request - skipping topic check")
                result = self.essay_chain.invoke({"query": full_query})
                print("LLM Result:", result)

                if not result:
                    raise ValueError("No response returned from the model.")

                return {
                    "status": "success",
                    "query": query,
                    "response": {
                        "questions": [
                            {
                                "question": result["question"],
                                "answer": result["answer"]
                            }
                        ]
                    },
                    "metadata": {
                        "model": self.model_name,
                        "rag": True,
                        "document_chunks": 4,
                        "type": "Essay",
                        "relevance_score": 1.0,
                        "skip_topic_check": True
                    }
                }

            topic_result = self.topic_chain.invoke({"query": full_query})
            print(f"Topic relevance check: {topic_result}")

            if not topic_result["is_relevant"] or topic_result["confidence"] < relevance_threshold:
                self._get_out_of_topic_response(query, topic_result["reason"])

            result = self.essay_chain.invoke({"query": full_query})
            print("LLM Result:", result)

            if not result:
                raise ValueError("No response returned from the model.")

            return {
                "status": "success",
                "query": query,
                "response": {
                    "questions": [
                        {
                            "question": result["question"],
                            "answer": result["answer"]
                        }
                    ]
                },
                "metadata": {
                    "model": self.model_name,
                    "rag": True,
                    "document_chunks": 4,
                    "type": "Essay",
                    "relevance_score": topic_result["confidence"]
                }
            }

        except Exception as e:
            print("ERROR:", e)
            if "context" in str(e).lower() or "relevant" in str(e).lower():
                self._get_out_of_topic_response(
                    query,
                    "Unable to process the query due to topic mismatch or insufficient context."
                )

            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "message": str(e)
                }
            )

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from uuid import uuid4
import json
from typing import Literal
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableMap
from fastapi import HTTPException  # <- Added

from core.config import OLLAMA_MODEL, OLLAMA_HOST
from core.dependencies import get_vector_retriever_en, get_vector_retriever


class MCQ(BaseModel):
    question: str = Field(description="A realistic and informative multiple-choice question.")
    A: str = Field(description="Option A for the multiple-choice question.")
    B: str = Field(description="Option B for the multiple-choice question.")
    C: str = Field(description="Option C for the multiple-choice question.")
    D: str = Field(description="Option D for the multiple-choice question.")
    answer: Literal['A', 'B', 'C', 'D'] = Field(description="The correct answer letter (A, B, C, or D).")
    explanation: str = Field(description="Explanation of why the correct answer is right.")


class TopicRelevance(BaseModel):
    is_relevant: bool = Field(description="Whether the query is relevant to the available context")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reason: str = Field(description="Brief explanation of the relevance assessment")


class MCQService:
    def __init__(self, language: str):
        self.model_name = OLLAMA_MODEL
        self.language = language

        self.model = ChatOllama(
            base_url=OLLAMA_HOST,
            model=OLLAMA_MODEL,
            temperature=0.7,
            options={
                "num_ctx": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "cache": False,
                "seed": -1
            }
        )

        self.parser = JsonOutputParser(pydantic_object=MCQ)
        self.topic_parser = JsonOutputParser(pydantic_object=TopicRelevance)

        topic_check_template_en = (
            "You are a topic relevance checker for a multiple-choice question generator.\n"
            "Consider the query relevant if:\n"
            "1. It asks to generate/create/make questions (always relevant)\n"
            "2. It's about a topic that can be found in the provided context\n"
            "3. It's a general request for questions/MCQs without specific topic (always relevant)\n\n"
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
            "Kamu adalah pemeriksa relevansi topik untuk generator soal pilihan ganda.\n"
            "Anggap pertanyaan relevan jika:\n"
            "1. Meminta untuk membuat/menghasilkan soal (selalu relevan)\n"
            "2. Tentang topik yang bisa ditemukan dalam konteks yang diberikan\n"
            "3. Permintaan umum untuk soal/MCQ tanpa topik spesifik (selalu relevan)\n\n"
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
            "You are a JSON API that returns a structured and high-quality multiple-choice question with answer and explanation.\n"
            "ONLY return a valid JSON object with REAL values. DO NOT include any placeholder text, example values, or descriptions.\n"
            "Use the following context to generate a meaningful and informative result.\n\n"
            "Context:\n{context}\n\n"
            "User query:\n{query}\n\n"
            "Respond ONLY in this exact JSON format:\n"
            "{{\n"
            '    "properties": {{\n'
            '        "question": "Your question here",\n'
            '        "A": "Option A",\n'
            '        "B": "Option B",\n'
            '        "C": "Option C",\n'
            '        "D": "Option D",\n'
            '        "answer": "Correct option letter (A/B/C/D)",\n'
            '        "explanation": "Explanation of the correct answer."\n'
            "    }}\n"
            "}}"
        )

        prompt_template_id = (
        "Kamu adalah sebuah API JSON yang menghasilkan soal pilihan ganda yang terstruktur dan berkualitas tinggi lengkap dengan jawaban dan penjelasan.\n"
        "HANYA kembalikan objek JSON yang valid dengan nilai yang SESUNGGUHNYA. JANGAN sertakan teks placeholder, contoh, atau deskripsi umum.\n"
        "Gunakan konteks berikut untuk menghasilkan pertanyaan yang bermakna dan informatif.\n\n"
        "Konteks:\n{context}\n\n"
        "Permintaan pengguna:\n{query}\n\n"
        "Balas HANYA dalam format JSON berikut ini:\n"
        "{{\n"
        '    "properties": {{\n'
        '        "question": "Pertanyaan di sini",\n'
        '        "A": "Pilihan A",\n'
        '        "B": "Pilihan B",\n'
        '        "C": "Pilihan C",\n'
        '        "D": "Pilihan D",\n'
        '        "answer": "Huruf pilihan benar (A/B/C/D)",\n'
        '        "explanation": "Penjelasan dari jawaban yang benar."\n'
        "    }}\n"
        "}}"
        )

        self.prompt = PromptTemplate(
            template=prompt_template_en if language == "english" else prompt_template_id,
            input_variables=["query", "context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        self.retriever = get_vector_retriever_en() if language == "english" else get_vector_retriever()

        self.topic_chain = RunnableMap({
            "context": lambda x: self.retriever.get_relevant_documents(x["query"]),
            "query": RunnablePassthrough()
        }) | self.topic_prompt | self.model | self.topic_parser

        self.mcq_chain = RunnableMap({
            "context": lambda x: self.retriever.get_relevant_documents(x["query"]),
            "query": RunnablePassthrough()
        }) | self.prompt | self.model | self.parser

    def _get_out_of_topic_response(self, query: str, reason: str = ""):
        if self.language == "english":
            message = f"I'm sorry, but your query '{query}' appears to be out of topic or not related to the available knowledge base. Please ask questions related to the topics I'm trained on."
            if reason:
                message += f" Reason: {reason}"
        else:
            message = f"Maaf, pertanyaan '{query}' tampaknya di luar topik atau tidak terkait dengan basis pengetahuan yang tersedia. Silakan ajukan pertanyaan yang berkaitan dengan topik yang saya kuasai."
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
                    "model": self.model_name,
                }
            }    
        )
        

    def _is_question_generation_request(self, query: str) -> bool:
        question_keywords = [
            "make", "create", "generate", "give me", "show me", "question", "questions", 
            "mcq", "multiple choice", "quiz", "test", "one question", "some questions",
            "buat", "buatkan", "bikin", "bikinkan", "kasih", "berikan", "soal", 
            "pertanyaan", "pilihan ganda", "kuis", "ujian", "satu soal"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in question_keywords)

    def run(self, query: str, relevance_threshold: float = 0.3):
        random_id = str(uuid4())[:8]
        full_query = f"({random_id}) {query}"

        try:
            if self._is_question_generation_request(query):
                print("Detected as question generation request - skipping topic check")
                mcq_result = self.mcq_chain.invoke({"query": full_query})
                print(f"MCQ result: {mcq_result}")

                return {
                    "status": "success",
                    "query": query,
                    "response": {
                        "questions": [
                            {
                                "questions": mcq_result["properties"]["question"],
                                "A": mcq_result["properties"]["A"],
                                "B": mcq_result["properties"]["B"],
                                "C": mcq_result["properties"]["C"],
                                "D": mcq_result["properties"]["D"],
                                "Answer": mcq_result["properties"]["answer"],
                                "Explanation": mcq_result["properties"]["explanation"],
                            }
                        ]
                    },
                    "metadata": {
                        "model": self.model_name,
                        "rag": True,
                        "type": "MCQ",
                        "relevance_score": 1.0, 
                        "skip_topic_check": True
                    }
                }

            topic_result = self.topic_chain.invoke({"query": full_query})
            print(f"Topic relevance check: {topic_result}")

            if not topic_result["is_relevant"] or topic_result["confidence"] < relevance_threshold:
                return self._get_out_of_topic_response(query, topic_result["reason"])

            mcq_result = self.mcq_chain.invoke({"query": full_query})
            print(f"MCQ result: {mcq_result}")

            return {
                "status": "success",
                "query": query,
                "response": {
                    "questions": [
                        {
                            "questions": mcq_result["properties"]["question"],
                            "A": mcq_result["properties"]["A"],
                            "B": mcq_result["properties"]["B"],
                            "C": mcq_result["properties"]["C"],
                            "D": mcq_result["properties"]["D"],
                            "Answer": mcq_result["properties"]["answer"],
                            "Explanation": mcq_result["properties"]["explanation"],
                        }
                    ]
                },
                "metadata": {
                    "model": self.model_name,
                    "rag": True,
                    "type": "MCQ",
                    "relevance_score": topic_result["confidence"]
                }
            }

        except Exception as e:
            print(f"Error in MCQ generation: {str(e)}")
            
            return self._get_out_of_topic_response(
                query, 
                "Unable to process the query due to technical issues or topic mismatch."
            )
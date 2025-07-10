from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from uuid import uuid4
from typing import Literal
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableMap

from core.config import OLLAMA_MODEL, OLLAMA_HOST
from core.dependencies import get_vector_retriever_en, get_vector_retriever


class MCQ(BaseModel):
    question: str = Field(
        description="A realistic and informative multiple-choice question.")
    A: str = Field(description="Option A for the multiple-choice question.")
    B: str = Field(description="Option B for the multiple-choice question.")
    C: str = Field(description="Option C for the multiple-choice question.")
    D: str = Field(description="Option D for the multiple-choice question.")
    answer: Literal['A', 'B', 'C', 'D'] = Field(
        description="The correct answer letter (A, B, C, or D).")
    explanation: str = Field(
        description="Explanation of why the correct answer is right.")


class TopicRelevance(BaseModel):
    is_relevant: bool = Field(
        description="Whether the query is relevant to the available context")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reason: str = Field(
        description="Brief explanation of the relevance assessment")


class MCQService:
    def __init__(self, language: str):
        self.language = language
        self.model_name = OLLAMA_MODEL

        self.model = ChatOllama(
            base_url=OLLAMA_HOST,
            model=self.model_name,
            options={
                "temperature": 0.1,
                "top_p": 0.40,
                "top_k": 50,
                "num_ctx": 4096,
                "cache": False,
                "seed": 42,
                "repeat_penalty": 1.1,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.2,
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
            "You are a JSON API that returns a concise, clear, and high-quality multiple-choice question (MCQ) with answer and explanation.\n"
            "ONLY return a valid JSON object with REAL values. DO NOT include any placeholder text, example values, or descriptions.\n"
            "Keep the question and each option as short and direct as possible, avoiding unnecessary details or repetition.\n"
            "Use the following context to generate a meaningful and informative result.\n\n"
            "Context:\n{context}\n\n"
            "User query:\n{query}\n\n"
            "Respond ONLY in this exact JSON format:\n"
            "{{\n"
            '    "properties": {{\n'
            '        "question": "Short, clear question",\n'
            '        "A": "Option A",\n'
            '        "B": "Option B",\n'
            '        "C": "Option C",\n'
            '        "D": "Option D",\n'
            '        "answer": "Correct option letter (A/B/C/D)",\n'
            '        "explanation": "Brief explanation of the correct answer."\n'
            "    }}\n"
            "}}"
        )

        prompt_template_id = (
            "Kamu adalah API JSON yang menghasilkan soal pilihan ganda (MCQ) yang singkat, jelas, dan berkualitas tinggi beserta jawaban dan penjelasan.\n"
            "HANYA kembalikan objek JSON valid dengan nilai SESUNGGUHNYA. JANGAN sertakan teks placeholder, contoh, atau deskripsi umum.\n"
            "Buat pertanyaan dan setiap opsi sependek dan sejelas mungkin, hindari detail atau pengulangan yang tidak perlu.\n"
            "Pastikan answer atau jawaban adalah huruf dari pilihan yang benar (A/B/C/D)\n"
            "Gunakan konteks berikut untuk menghasilkan hasil yang bermakna dan informatif.\n\n"
            "Konteks:\n{context}\n\n"
            "Permintaan pengguna:\n{query}\n\n"
            "Balas HANYA dalam format JSON berikut ini:\n"
            "{{\n"
            '    "properties": {{\n'
            '        "question": "Pertanyaan singkat dan jelas",\n'
            '        "A": "Pilihan A",\n'
            '        "B": "Pilihan B",\n'
            '        "C": "Pilihan C",\n'
            '        "D": "Pilihan D",\n'
            '        "answer": "Huruf pilihan benar (A/B/C/D)",\n'
            '        "explanation": "Penjelasan singkat dari jawaban yang benar."\n'
            "    }}\n"
            "}}"
        )

        self.prompt = PromptTemplate(
            template=prompt_template_en if language == "english" else prompt_template_id,
            input_variables=["query", "context"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()},
        )

        self.retriever = get_vector_retriever_en(
        ) if language == "english" else get_vector_retriever()

        self.topic_chain = RunnableMap({
            "context": lambda x: self.retriever.get_relevant_documents(x["query"]),
            "query": RunnablePassthrough()
        }) | self.topic_prompt | self.model | self.topic_parser

        self.mcq_chain = RunnableMap({
            "context": lambda x: self.retriever.get_relevant_documents(x["query"]),
            "query": RunnablePassthrough()
        }) | self.prompt | self.model | self.parser

    def _get_out_of_topic_result(self, query: str, reason: str = ""):
        if self.language == "english":
            explanation = reason or "Your query is not relevant to the available topics."
            question_text = "Your query appears to be out of topic or not related to our knowledge base."
        else:
            explanation = reason or "Pertanyaan Anda tidak relevan dengan topik yang tersedia."
            question_text = "Pertanyaan Anda tampaknya di luar topik atau tidak terkait dengan basis pengetahuan kami."

        return {
            "status": "success",
            "query": query,
            "response": {
                "questions": [
                    {
                        "question": question_text,
                        "A": "-",
                        "B": "-",
                        "C": "-",
                        "D": "-",
                        "answer": "-",
                        # "Explanation": explanation,
                    }
                ]
            },
            "metadata": {
                "model": self.model_name,
                "rag": False,
                "type": "MCQ",
                "relevance_score": 0.0,
                "reason": explanation,
                "skip_topic_check": False
            }
        }

    def _is_question_generation_request(self, query: str) -> bool:
        keywords = [
            "make", "create", "generate", "give me", "show me", "question", "questions",
            "mcq", "multiple choice", "quiz", "test", "one question", "some questions",
            "buat", "buatkan", "bikin", "bikinkan", "kasih", "berikan", "soal",
            "pertanyaan", "pilihan ganda", "kuis", "ujian", "satu soal"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in keywords)

    def run(self, query: str, relevance_threshold: float = 0.3):
        random_id = str(uuid4())[:8]
        full_query = f"({random_id}) {query}"

        try:
            if self._is_question_generation_request(query):
                print("Detected as question generation request - skipping topic check")
                mcq_result = self.mcq_chain.invoke({"query": full_query})
                print(f"MCQ generation result: {mcq_result}")
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
                                "answer": mcq_result["properties"]["answer"],
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
                return self._get_out_of_topic_result(query, topic_result["reason"])

            mcq_result = self.mcq_chain.invoke({"query": full_query})
            return {
                "status": "success",
                "query": query,
                "response": {
                    "questions": [
                        {
                            "question": mcq_result["properties"]["question"],
                            "A": mcq_result["properties"]["A"],
                            "B": mcq_result["properties"]["B"],
                            "C": mcq_result["properties"]["C"],
                            "D": mcq_result["properties"]["D"],
                            "answer": mcq_result["properties"]["answer"],
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
            return self._get_out_of_topic_result(query, "An internal error occurred or topic mismatch.")

from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from uuid import uuid4
from typing import Literal
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableMap
from better_profanity import profanity

from core.config import VLLM_CHAT_MODEL, VLLM_CHAT_URL
from core.dependencies import get_vector_retriever_en, get_vector_retriever


class MCQ(BaseModel):
    question: str = Field(description="MCQ question")
    A: str = Field(description="Option A")
    B: str = Field(description="Option B")
    C: str = Field(description="Option C")
    D: str = Field(description="Option D")
    Answer: Literal['A', 'B', 'C', 'D'] = Field(description="Correct option letter")
    explanation: str = Field(description="Explanation of the correct Answer")


class TopicRelevance(BaseModel):
    is_relevant: bool = Field(description="Whether query is relevant")
    confidence: float = Field(description="Confidence 0.0–1.0")
    reason: str = Field(description="Explanation of the relevance check")


class MCQService:
    def __init__(self, language: str):
        self.language = language
        self.model_name = VLLM_CHAT_MODEL
        self.model = ChatOpenAI(
            model=self.model_name,
            api_key="not-needed",  
            base_url=VLLM_CHAT_URL,
            temperature=0.1,
            max_tokens=None,
            model_kwargs={
                "top_p": 0.40,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.2,
                "seed": 42,
            },
        )

        self.parser = JsonOutputParser(pydantic_object=MCQ)
        self.topic_parser = JsonOutputParser(pydantic_object=TopicRelevance)

        topic_template = (
            "You are a topic relevance checker for a multiple-choice question generator.\n"
            "Consider relevant if:\n"
            "1. Query explicitly asks for generating questions (always relevant)\n"
            "2. Topic exists in provided context\n"
            "3. Query is a general request for questions (always relevant)\n\n"
            "Context:\n{context}\n\n"
            "User query: {query}\n\n"
            "Respond ONLY as JSON:\n"
            "{{\"is_relevant\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"...\"}}"
            if language == "english"
            else
            "Kamu adalah pemeriksa relevansi topik soal pilihan ganda.\n"
            "Anggap relevan jika:\n"
            "1. Diminta membuat soal (selalu relevan)\n"
            "2. Topik ada dalam konteks\n"
            "3. Permintaan umum soal (selalu relevan)\n\n"
            "Konteks:\n{context}\n\n"
            "Pertanyaan pengguna: {query}\n\n"
            "Jawab HANYA dalam JSON:\n"
            "{{\"is_relevant\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"...\"}}"
        )
        self.topic_prompt = PromptTemplate(
            template=topic_template,
            input_variables=["query", "context"]
        )

        mcq_template = (
            "You are a JSON API returning a clear MCQ with options and explanation.\n"
            "ONLY return valid JSON (no placeholders).\n\n"
            "Context:\n{context}\n\n"
            "User query:\n{query}\n\n"
            "Respond ONLY in this JSON:\n"
            "{{\"question\": \"...\", \"A\": \"...\", \"B\": \"...\", \"C\": \"...\", \"D\": \"...\", \"Answer\": \"A/B/C/D\", \"explanation\": \"...\"}}"
            if language == "english"
            else
            "Kamu adalah API JSON yang menghasilkan soal pilihan ganda singkat dan jelas.\n"
            "Hanya kembalikan JSON valid tanpa placeholder.\n\n"
            "Konteks:\n{context}\n\n"
            "Pertanyaan pengguna:\n{query}\n\n"
            "Balas HANYA dalam JSON ini:\n"
            "{{\"question\": \"...\", \"A\": \"...\", \"B\": \"...\", \"C\": \"...\", \"D\": \"...\", \"Answer\": \"A/B/C/D\", \"explanation\": \"...\"}}"
        )
        self.prompt = PromptTemplate(
            template=mcq_template,
            input_variables=["query", "context"]
        )

        self.retriever = get_vector_retriever_en() if language == "english" else get_vector_retriever()

        self.topic_chain = (
            RunnableMap({
                "context": lambda x: self.retriever.get_relevant_documents(x["query"]),
                "query": RunnablePassthrough()
            })
            | self.topic_prompt
            | self.model
            | self.topic_parser
        )
        self.mcq_chain = (
            RunnableMap({
                "context": lambda x: self.retriever.get_relevant_documents(x["query"]),
                "query": RunnablePassthrough()
            })
            | self.prompt
            | self.model
            | self.parser
        )

    def _get_out_of_topic_result(self, query: str, reason: str = ""):
        explanation = reason or (
            "Your query is not relevant to available topics."
            if self.language == "english"
            else "Pertanyaan Anda tidak relevan dengan topik yang tersedia."
        )
        return {
            "status": "success",
            "query": query,
            "response": {
                "questions": [{
                    "question": explanation,
                    "A": "-",
                    "B": "-",
                    "C": "-",
                    "D": "-",
                    "Answer": "-",
                    # "explanation": explanation
                }]
            },
            "metadata": {
                "model": self.model_name,
                "rag": False,
                "type": "MCQ",
                "relevance_score": 0.0,
                "reason": explanation
            }
        }

    def _is_question_generation_request(self, query: str) -> bool:
        keywords = [
            "make", "create", "generate", "give me", "question", "questions",
            "mcq", "multiple choice", "quiz", "test",
            "buat", "bikin", "kasih", "soal", "pertanyaan",
            "pilihan ganda", "kuis", "ujian"
        ]
        return any(k in query.lower() for k in keywords)

    def run(self, query: str, relevance_threshold: float = 0.3):
        profanity.load_censor_words(["anjing", "kontol", "memek", "tolol", "goblok", "idiot"])

        if profanity.contains_profanity(query):
            return {"status": "error", "query": query, "message": "Inappropriate language detected."}

        random_id = str(uuid4())[:8]
        full_query = f"({random_id}) {query}"

        try:
            if self._is_question_generation_request(query):
                mcq = self.mcq_chain.invoke({"query": full_query})
                return {
                    "status": "success",
                    "query": query,
                    "response": {"questions": [mcq]}, 
                    "metadata": {
                        "model": self.model_name,
                        "rag": True,
                        "type": "MCQ",
                        "relevance_score": 1.0,
                        "reason": "Direct question-generation request"
                    }
                }

            topic = self.topic_chain.invoke({"query": full_query})
            if not topic["is_relevant"] or topic["confidence"] < relevance_threshold:  # Access as dict
                return self._get_out_of_topic_result(query, topic["reason"])

            mcq = self.mcq_chain.invoke({"query": full_query})
            return {
                "status": "success",
                "query": query,
                "response": {"questions": [mcq]},  
                "metadata": {
                    "model": self.model_name,
                    "rag": True,
                    "type": "MCQ",
                    "relevance_score": topic["confidence"],  # Access as dict
                    # "reason": topic["reason"]  # Access as dict
                }
            }

        except Exception as e:
            return self._get_out_of_topic_result(query, f"Internal error: {str(e)}")
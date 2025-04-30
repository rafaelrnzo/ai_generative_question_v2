from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import json
from uuid import uuid4
from core.config import OLLAMA_MODEL, OLLAMA_HOST
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableMap
from typing import Optional
from core.dependencies import get_vector_retriever_en, get_vector_retriever


class Essay(BaseModel):
    question: str = Field(description="A realistic and informative essay question.")
    answer: str = Field(description="A concise answer to the question.")


class EssayService:
    def __init__(self, language: str):
        self.model_name = OLLAMA_MODEL

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

        self.parser = JsonOutputParser(pydantic_object=Essay)

        prompt_template_en = (
            "You are a JSON-only API that returns one high-quality essay question and its answer.\n"
            "Respond with ONLY a valid JSON object, no explanations, no extra text. All values must be REAL and COMPLETE.\n\n"
            "Use the following context to generate a meaningful question and answer:\n"
            "Context:\n{context}\n\n"
            "User query: {query}\n\n"
            "Return exactly in the following JSON format:\n"
            '{{\n'
            '  "question": "A clear, standalone essay question based on the context.",\n'
            '  "answer": "A well-structured and informative answer to the question."\n'
            '}}'
        )

        prompt_template_id = (
            "Kamu adalah API JSON-only yang menghasilkan satu pertanyaan esai berkualitas tinggi dan jawabannya.\n"
            "Hanya berikan OBJEK JSON valid, tanpa penjelasan atau teks tambahan. Semua nilai harus NYATA dan LENGKAP.\n\n"
            "Gunakan konteks berikut untuk membuat pertanyaan dan jawaban yang bermakna:\n"
            "Konteks:\n{context}\n\n"
            "Permintaan pengguna: {query}\n\n"
            "Kembalikan dalam format JSON persis seperti ini:\n"
            '{{\n'
            '  "question": "Pertanyaan esai yang jelas dan berdiri sendiri berdasarkan konteks.",\n'
            '  "answer": "Jawaban yang terstruktur dan informatif untuk pertanyaan tersebut."\n'
            '}}'
        )

        self.prompt = PromptTemplate(
            template=prompt_template_en if language == "english" else prompt_template_id,
            input_variables=["query", "context"]
        )

        self.retriever = get_vector_retriever_en() if language == "english" else get_vector_retriever()

        self.chain = RunnableMap({
            "context": lambda x: self.retriever.get_relevant_documents(x["query"]),
            "query": lambda x: x["query"]
        }) | self.prompt | self.model | self.parser

    def run(self, query: str):
        random_id = str(uuid4())[:8]
        full_query = f"({random_id}) {query}"

        try:
            result = self.chain.invoke({"query": full_query})
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
                    "type": "Essay"
                }
            }

        except Exception as e:
            print("ERROR:", e)
            return {
                "status": "error",
                "message": str(e)
            }

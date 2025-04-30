from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from uuid import uuid4
import json
from typing import Literal
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableMap

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


class MCQService:
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

        self.parser = JsonOutputParser(pydantic_object=MCQ)

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

        self.chain = RunnableMap({
            "context": lambda x: self.retriever.get_relevant_documents(x["query"]),
            "query": RunnablePassthrough()
        }) | self.prompt | self.model | self.parser

    def run(self, query: str):
        random_id = str(uuid4())[:8]
        full_query = f"({random_id}) {query}"

        result = self.chain.invoke({"query": full_query})
        print(result)

        return {
            "status": "success",
            "query": query,
            "response": {
                "questions": [
                    {
                        "questions" : result["properties"]["question"],
                        "A" : result["properties"]["A"],
                        "B" : result["properties"]["B"],
                        "C" : result["properties"]["C"],
                        "D" : result["properties"]["D"],
                        "Answer" : result["properties"]["answer"],
                        "Explanation" : result["properties"]["explanation"],
                    }
                ]
            },
            "metadata": {
                "model": self.model_name,
                "rag": True,
                "type": "MCQ"
            }
        }
          

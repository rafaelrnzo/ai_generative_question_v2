import json
import re
import ollama
from fastapi import HTTPException, APIRouter, Depends
from fastapi.responses import JSONResponse
from core.config import OLLAMA_HOST, OLLAMA_MODEL

router = APIRouter()

class EssayService:
    def __init__(self):
        ollama.base_url = OLLAMA_HOST
        self.model = OLLAMA_MODEL
    
    @staticmethod
    def format_essay_prompt(question: str, context: str, num_questions=10) -> str:
        return f"""Anda adalah seorang **dosen berpengalaman dalam bidang {context}**.
            Buatlah **{num_questions} soal ESSAY** berdasarkan permintaan berikut:

            **Instruksi WAJIB untuk format output:**  
            - **Hanya buat soal ESSAY (bukan pilihan ganda).**  
            - **Setiap soal harus memiliki jawaban yang lengkap.**  
            - **Tidak boleh menggunakan huruf A, B, C, atau D dalam soal maupun jawaban.**  
            - **WAJIB menggunakan format berikut:**  

            **FORMAT OUTPUT:**
            Soal 1:
            [Isi pertanyaan ESSAY, buatkan pertanyaan nya secara mendetail, kompleks, dan juga sesuai dengan studi kasus yang ada di dalam data]
            Jawaban: [Isi jawaban ESSAY yang lengkap]

            Soal 2:
            [Isi pertanyaan ESSAY, buatkan pertanyaan nya secara mendetail, kompleks, dan juga sesuai dengan studi kasus yang ada di dalam data]
            Jawaban: [Isi jawaban ESSAY yang lengkap]

            **Jangan buat soal pilihan ganda.**  
            **Pastikan ada "Jawaban" untuk setiap "Soal".**  
            **Jangan gunakan format pilihan ganda (A, B, C, D) dalam bentuk apapun.**  

            **Konteks:** {context}  
            **Jumlah soal:** {num_questions}  
            **Permintaan pengguna:** {question}
        """

    def generate_essay(self, question: str, context: str):
        num_questions = 10  
        num_match = re.search(r'(\d+)\s*(?:soal|pertanyaan|question)', question, re.IGNORECASE)
        if num_match:
            num_questions = int(num_match.group(1))
            print(f"Detected request for {num_questions} questions")
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user', 
                    'content': self.format_essay_prompt(question, context, num_questions)
                }]
            )
            
            content = response['message']['content']
            print(f"LLM Response (first 300 chars):\n{content[:300]}")  # Debugging

            parsed_json = self.parse_essay_text(content, num_questions)

            if parsed_json["total_questions"] < num_questions:
                print("Warning: Incomplete questions extracted. Retrying with a modified prompt...")
                new_prompt = self.format_essay_prompt(question, context, num_questions)
                response = ollama.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': new_prompt}]
                )
                content = response['message']['content']
                parsed_json = self.parse_essay_text(content, num_questions)

            return parsed_json
            
        except Exception as e:
            import traceback
            print(f"Error in generate_essay: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error processing LLM response: {str(e)}")
    
    @staticmethod
    def parse_essay_text(content: str, expected_count=10):
        questions = []
        
        matches = re.findall(
            r'(?:^|\n)(?:\d+\.|\*\*Soal \d+:\*\*)\s*(.*?)(?:\n\n(?:Jawaban:|Jawaban)|\n\n\d+\.|\n\n\*\*Soal|\Z)', 
            content, 
            re.DOTALL | re.MULTILINE
        )

        answer_matches = re.findall(
            r'(?:Jawaban:|Jawaban)\s*(.*?)(?=\n\n\d+\.|\n\n\*\*Soal|\Z)', 
            content, 
            re.DOTALL | re.MULTILINE
        )

        for i in range(min(len(matches), len(answer_matches))):
            questions.append({
                "number": i + 1,
                "question": matches[i].strip(),
                "answer": answer_matches[i].strip()
            })
        
        if len(questions) < expected_count:
            print(f"Warning: Only {len(questions)} out of {expected_count} questions extracted!")
            print("Raw content sample:\n", content[:1000])  # Show first 1000 chars

        return {
            "total_questions": len(questions),
            "questions": questions
        }
        
    def generate_json_response(self, question: str, context: str):
        return self.generate_essay(question, context)
import json
import re
import ollama
from fastapi import HTTPException
from utils.mcq_json import parse_mcq_text
from core.config import OLLAMA_HOST, OLLAMA_MODEL

class LLMService:
    def __init__(self):
        ollama.base_url = OLLAMA_HOST
        self.model = OLLAMA_MODEL

    def format_mcq_prompt(self, question: str, context: str, num_questions=1, language='indonesian') -> str:
        if language.lower() == "indonesian":
            return f"""Anda adalah seorang **dosen berpengalaman dalam bidang {context}**. 
            Tugas Anda adalah membuat **{num_questions} soal** pilihan ganda dengan kualitas tinggi. 

            **Instruksi SANGAT PENTING untuk format output:**
            - Gunakan bahasa yang jelas dan tidak ambigu.
            - Setiap soal HARUS memiliki empat pilihan jawaban A, B, C, dan D.
            - Jawaban benar harus tersebar secara acak di antara A, B, C, atau D.
            - Setiap soal HARUS diikuti oleh baris "Jawaban: X" di mana X adalah pilihan yang benar (A, B, C, atau D).
            - Format ini HARUS konsisten untuk SEMUA {num_questions} soal.
            - SELALU tulis semua {num_questions} soal yang diminta.

            **Format yang HARUS DIIKUTI:**

            Soal 1:
            [Pertanyaan lengkap]  
            A) [Pilihan A]
            B) [Pilihan B]  
            C) [Pilihan C]  
            D) [Pilihan D]  
            Jawaban: [A/B/C/D]

            Soal 2:
            [Pertanyaan lengkap]  
            A) [Pilihan A]  
            B) [Pilihan B]  
            C) [Pilihan C]  
            D) [Pilihan D]  
            Jawaban: [A/B/C/D]

            [Dan seterusnya sampai Soal {num_questions}]

            **PENTING:** Pastikan setiap soal memiliki jawaban yang jelas dengan format "Jawaban: X" tepat setelah pilihan D.
            Jika tidak mengikuti format ini dengan tepat, sistem tidak akan dapat memproses jawaban dengan benar.

            **Konteks:** {context}  
            **Permintaan:** {question}  
            **Jumlah soal yang harus dibuat: {num_questions}**
            """

        elif language.lower() == "english":
            return f"""You are an **experienced lecturer in the field of {context}**. 
            Your task is to create **{num_questions} high-quality multiple-choice questions**.

            **VERY IMPORTANT Instructions for output format:**
            - Use clear and unambiguous language.
            - Each question MUST have four answer options: A, B, C, and D.
            - The correct answers must be randomly distributed among A, B, C, and D.
            - Each question MUST be followed by a line that says "Answer: X" where X is the correct choice (A, B, C, or D).
            - This format MUST be consistent for ALL {num_questions} questions.
            - ALWAYS write exactly {num_questions} questions as requested.

            **MANDATORY Format:**

            **Question 1:**
            [Full question]  
            A) [Option A]  
            B) [Option B]  
            C) [Option C]  
            D) [Option D]  
            Answer: [A/B/C/D]

            **Question 2:**
            [Full question]  
            A) [Option A]  
            B) [Option B]  
            C) [Option C]  
            D) [Option D]  
            Answer: [A/B/C/D]

            [And so on until Question {num_questions}]

            **IMPORTANT:** Make sure each question has a clearly defined answer in the format "Answer: X" immediately after option D.  
            If you do not follow this format precisely, the system will not be able to process the answers correctly.

            **Context:** {context}  
            **Task:** {question}  
            **Number of questions to generate: {num_questions}**
            """

    @staticmethod
    def enhance_content_format(content: str) -> str: 
        lines = content.split('\n')
        enhanced_lines, current_options = [], {}
        is_english = "Question" in content or "Answer:" in content
        answer_marker = "Answer:" if is_english else "Jawaban:"
        question_pattern = r'(?:\*\*)?(?:Question|Soal)\s+(\d+)(?:\*\*)?[:\.]?'

        option_count = 0
        for line in lines:
            stripped = line.strip()
            if re.search(question_pattern, stripped):
                if option_count == 4 and not any(answer_marker in l for l in enhanced_lines[-5:]):
                    enhanced_lines.append(f"{answer_marker} A")
                option_count = 0
                enhanced_lines.append(line)
                continue

            if re.match(r'^([A-D])[\s\)\.:]+\s*(.*)', stripped):
                option_count += 1
                enhanced_lines.append(line)
                continue

            if answer_marker in stripped:
                enhanced_lines.append(line)
                continue

            if option_count == 4 and not any(answer_marker in l for l in enhanced_lines[-4:]):
                enhanced_lines.append(f"{answer_marker} A")

            enhanced_lines.append(line)

        if option_count == 4 and not any(answer_marker in l for l in enhanced_lines[-5:]):
            enhanced_lines.append(f"{answer_marker} A")

        return '\n'.join(enhanced_lines)

    def generate_mcq(self, question: str, language: str, context: str):
        try:
            language = language.lower() if language else "indonesian"
            num_questions = 10

            num_match = re.search(r'(\d+)\s*(soal|pertanyaan|question)', question, re.IGNORECASE)
            if num_match:
                num_questions = int(num_match.group(1))

            prompt = self.format_mcq_prompt(question, context, num_questions, language)
            response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            content = response['message']['content']
            parsed_json = parse_mcq_text(content)

            if parsed_json["total_questions"] < num_questions:
                enhanced_content = self.enhance_content_format(content)
                parsed_json = parse_mcq_text(enhanced_content)

            return json.loads(json.dumps(parsed_json))

        except Exception as e:
            import traceback
            print(f"Error in generate_mcq: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error processing LLM response: {str(e)}")

    def generate_json_response(self, question: str, language: str, context: str, num_questions: int = 1):
        try:
            prompt = self.format_mcq_prompt(question, context, num_questions, language)
            response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            return {
                "response": response['message']['content'],
                "type": "general_query"
            }
        except Exception as e:
            print(f"Error in generate_json_response: {str(e)}")
            return None

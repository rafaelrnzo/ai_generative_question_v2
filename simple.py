import json
import re
import ollama
from fastapi import HTTPException, APIRouter
from core.config import OLLAMA_HOST, OLLAMA_MODEL

router = APIRouter()

class EssayService:
    def __init__(self):
        ollama.base_url = OLLAMA_HOST
        self.model = OLLAMA_MODEL
    
    def format_essay_prompt(self, question: str, context: str, num_questions=1, language='indonesian') -> str:
        if language.lower() == "indonesian":
            return f"""Anda adalah dosen bidang {context}.
            Buatlah {num_questions} soal ESSAY berdasarkan: {question}

            Instruksi penting:
            - Buat HANYA soal essay dengan pertanyaan terbuka
            - DILARANG membuat soal pilihan ganda atau opsi A), B), C), D)
            - Setiap soal harus memiliki jawaban yang lengkap
            - Maksimal soal adalah 1 yang dibuat
            - Buatkan pertanyaanya secara acak, sehingga tidak akan terjadi pengulangan response yang sama, saya ingin respon nya unik.

            Format output:
            [Pertanyaan essay]
            Jawaban: [Jawaban lengkap]

            [Pertanyaan essay]
            Jawaban: [Jawaban lengkap]
            
            catatan: hanya sertakan format yang terkait, mohon untuk tidak berikan deskripsi tambahan terkait dengan response yang diberikan
            """
        else:
            return f"""You are a professor in {context}.
            Create {num_questions} ESSAY questions based on: {question}

            Important instructions:
            - Create ONLY essay questions with open-ended format
            - DO NOT create multiple choice questions or options A), B), C), D)
            - Each question must have a complete answer
            - The maximal response of question is 1, don't write more than 1.
            - Make the question random, so the responses not repetitively, I want the response unique.

            Output format:
            [Essay question]
            Answer: [Complete answer]

            [Essay question]
            Answer: [Complete answer]

            Note: please just give the response like the format, don't give another description, just throw it.
            """

    def generate_essay(self, question: str, language: str, context: str):
        num_questions = 1
        language = language.lower() if language else "indonesian"
        
        num_match = re.search(r'(\d+)\s*(?:soal|pertanyaan|question)', question, re.IGNORECASE)
        if num_match:
            num_questions = int(num_match.group(1))
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': self.format_essay_prompt(question, context, num_questions, language)}]
            )
            content = response['message']['content']
            
            if re.search(r'[A-D]\)', content) or re.search(r'[A-D]\s*\)', content):
                retry_prompt = self._get_retry_prompt(language, num_questions, context)
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {'role': 'user', 'content': self.format_essay_prompt(question, context, num_questions, language)},
                        {'role': 'assistant', 'content': content},
                        {'role': 'user', 'content': retry_prompt}
                    ]
                )
                content = response['message']['content']

            parsed_json = self.parse_essay_text(content, num_questions, language)

            if parsed_json["total_questions"] < num_questions:
                completion_prompt = self._get_completion_prompt(language, num_questions, parsed_json["total_questions"], context)
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {'role': 'user', 'content': self.format_essay_prompt(question, context, num_questions, language)},
                        {'role': 'assistant', 'content': content},
                        {'role': 'user', 'content': completion_prompt}
                    ]
                )
                content = response['message']['content']
                parsed_json = self.parse_essay_text(content, num_questions, language)

            self.clean_multiple_choice_format(parsed_json, language)
            
            return parsed_json
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing LLM response: {str(e)}")
    
    def _get_retry_prompt(self, language, num_questions, context):
        if language == "indonesian":
            return f"PERHATIAN: Jangan buat pilihan ganda. Saya HANYA butuh {num_questions} soal ESSAY tanpa opsi A/B/C/D."
        else:
            return f"ATTENTION: Do not create multiple choice. I ONLY need {num_questions} ESSAY questions without A/B/C/D options."
    
    def _get_completion_prompt(self, language, expected, actual, context):
        if language == "indonesian":
               return f"Saya perlu tepat {expected} soal essay tentang {context}. Jawaban sebelumnya hanya berisi {actual} soal."
        else:
            return f"I need exactly {expected} essay questions about {context}. Your previous answer only contained {actual} questions."
    
    @staticmethod
    def clean_multiple_choice_format(parsed_json, language='indonesian'):
        for q in parsed_json["questions"]:
            q["question"] = re.sub(r'\n[A-D]\)[^\n]+', '', q["question"])
            q["answer"] = re.sub(r'\n[A-D]\)[^\n]+', '', q["answer"])
            
            if len(q["question"].strip()) < 100 and not q["question"].strip().endswith('?'):
                if language.lower() == "indonesian":
                    q["question"] = f"Jelaskan secara detail tentang {q['question'].strip()}?"
                else:
                    q["question"] = f"Explain in detail about {q['question'].strip()}?"

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'[\*\-\•]\s*', '', text)           
        text = re.sub(r'\n+', ' ', text)                  
        text = re.sub(r'\s{2,}', ' ', text).strip()
        return text

    @staticmethod
    def parse_essay_text(content: str, expected_count=1, language='indonesian'):
        questions = []
        
        question_label = "Soal" if language == "indonesian" else "Question"
        answer_label = "Jawaban" if language == "indonesian" else "Answer"
        
        pattern = fr'(?:{question_label}\s*(\d+):?|(?<!\w)(\d+)\.)\s*(.*?)(?:\n+(?:{answer_label}:?|{answer_label}\s*\d+:?)\s*(.*?)(?=\n+(?:{question_label}\s*\d+:|(?<!\w)\d+\.)|$))'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if matches:
            for i, match in enumerate(matches):
                questions.append({
                    "number": i + 1,
                    "question": EssayService.clean_text(match[2]),
                    "answer": EssayService.clean_text(match[3])
                })
        else:
            question_pattern = fr'(?:^|\n)(?:{question_label}\s*\d+:?|(?<!\w)\d+\.)?\s*(.*?)(?=\n+(?:{answer_label}:?|{answer_label}\s*\d+:?))'
            answer_pattern = fr'(?:{answer_label}:?|{answer_label}\s*\d+:?)\s*(.*?)(?=\n+(?:{question_label}\s*\d+:|(?<!\w)\d+\.)|$)'
            
            question_matches = re.findall(question_pattern, content, re.DOTALL)
            answer_matches = re.findall(answer_pattern, content, re.DOTALL)
            
            for i in range(min(len(question_matches), len(answer_matches))):
                questions.append({
                    "number": i + 1,
                    "question": EssayService.clean_text(question_matches[i]),
                    "answer": EssayService.clean_text(answer_matches[i])
                })

        return {
            "total_questions": len(questions),
            "questions": questions
        }
        
    def generate_json_response(self, question: str, language: str, context: str):
        return self.generate_essay(question, language, context)


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
            return f"""Anda adalah seorang profesional berpengalaman di bidang **{context}**.  
            Tugas Anda adalah membuat **1 soal pilihan ganda (MCQ)** yang **unik, kreatif**, dan **tidak boleh berulang**, berdasarkan topik berikut:

            📌 **Topik / Permintaan:** {question}

            🎯 **Petunjuk penting yang HARUS diikuti:**
            1. Soal yang dibuat harus **berbeda setiap kali** (tidak boleh mengulang atau terlalu mirip).
            2. Soal harus **berdasarkan fakta yang benar** dan relevan dengan topik.
            3. Gunakan **variasi dalam kata-kata, fokus, dan sudut pandang** agar tidak repetitif.
            4. Jawaban benar harus **acak secara konsisten** di antara A, B, C, atau D.
            5. Jangan sertakan penjelasan, deskripsi, atau ucapan apa pun — cukup soal, opsi, dan jawaban.

            ✅ **Format WAJIB (ikuti persis seperti ini):**

            [Soal lengkap]  
            A) [Pilihan A]  
            B) [Pilihan B]  
            C) [Pilihan C]  
            D) [Pilihan D]  
            Jawaban: [A/B/C/D] ← HARUS ADA!

            🚫 **Catatan:**
            - Jangan menambahkan penjelasan, catatan kaki, atau konteks tambahan apa pun.
            - Jika Anda tidak mengikuti format ini, maka soal akan ditolak.
            """

        elif language.lower() == "english":
            return f"""You are an **expert professor** in the subject of **{context}**.  
            Your job is to create **one unique, creative, and non-repetitive multiple-choice question (MCQ)** on the following topic:

            **Topic / Task:** {question}

            Please ensure:
            1. The question is **not generic** or repeated from earlier knowledge.
            2. The MCQ should test a **specific fact or concept** from the topic.
            3. The wording, focus, or framing of the question should be different every time this is called.
            4. Randomize the correct answer between A, B, C, or D.

            **You MUST follow this format exactly**:

            Question 1:  
            [Full question]  
            A) [Option A]  
            B) [Option B]  
            C) [Option C]  
            D) [Option D]  
            Answer: [A/B/C/D] ← REQUIRED

            Only give the question, options, and the answer. Do NOT add anything else.
            """

    @staticmethod
    def enhance_content_format(content: str) -> str: 
        lines = content.split('\n')
        enhanced_lines = []
        option_count = 0
        
        is_english = any(term in content for term in ["Question", "Answer:", "Correct", "The answer is"])
        answer_marker = "Answer:" if is_english else "Jawaban:"
        
        question_pattern = r'(?:\*\*)?(?:Question|Soal)\s+(\d+)(?:\*\*)?[:\.]?'
        alt_question_pattern = r'^\s*(\d+)[\.\:]\s*(.*)$'

        inside_question = False
        has_seen_answer = False
        current_options = []

        for line in lines:
            stripped = line.strip()

            if re.search(question_pattern, stripped) or re.search(alt_question_pattern, stripped):
                if option_count > 0 and not has_seen_answer:
                    enhanced_lines.append(f"{answer_marker} A")
                inside_question = True
                option_count = 0
                has_seen_answer = False
                current_options = []
                enhanced_lines.append(line)
                continue

            option_match = re.match(r'^([A-D])[\s\)\.:]+\s*(.*)', stripped)
            if option_match:
                option_count += 1
                label = option_match.group(1)
                text = option_match.group(2).strip()
                # Remove trailing noise
                text = re.split(r'(?i)(let me know|adjust the difficulty|generate another question)', text)[0].strip()
                current_options.append(label)
                enhanced_lines.append(f"{label}) {text}")
                continue

            answer_patterns = [
                r'(?:[Jj]awaban|[Aa]nswer)[\s\:\=]+([A-D])',
                r'[Cc]orrect\s+[Aa]nswer[\s\:\=]+([A-D])',
                r'[Cc]orrect[\s\:\=]+([A-D])',
                r'[Tt]he\s+[Aa]nswer\s+is[\s\:\=]+([A-D])',
                r'[Kk]ey[\s\:\=]+([A-D])'
            ]

            for pattern in answer_patterns:
                match = re.search(pattern, stripped)
                if match:
                    answer = match.group(1).strip().upper()
                    if answer in ['A', 'B', 'C', 'D']:
                        enhanced_lines.append(f"{answer_marker} {answer}")
                        has_seen_answer = True
                        break
            else:
                # If 4 options seen but no answer, add fallback
                if option_count == 4 and not has_seen_answer and all(opt in current_options for opt in ['A', 'B', 'C', 'D']):
                    enhanced_lines.append(f"{answer_marker} A")
                    has_seen_answer = True

            enhanced_lines.append(line)

        # Final fallback
        if option_count > 0 and not has_seen_answer:
            enhanced_lines.append(f"{answer_marker} A")

        return '\n'.join(enhanced_lines)

    def generate_mcq(self, question: str, language: str, context: str):
        try:
            language = language.lower() if language else "indonesian"
            num_questions = 1

            num_match = re.search(r'(\d+)\s*(soal|pertanyaan|question)', question, re.IGNORECASE)
            if num_match:
                num_questions = int(num_match.group(1))

            prompt = self.format_mcq_prompt(question, context, num_questions, language)
            response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            content = response['message']['content']
            
            print("\n---------- RAW LLM RESPONSE ----------")
            print(content)
            print("--------------------------------------\n")
            
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

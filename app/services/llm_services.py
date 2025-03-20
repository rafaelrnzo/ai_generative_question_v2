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
    
    @staticmethod
    def format_mcq_prompt(question: str, context: str, num_questions=10) -> str:
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
        
        **Soal 1:**
        [Pertanyaan lengkap]
        A) [Pilihan A]
        B) [Pilihan B]
        C) [Pilihan C]
        D) [Pilihan D]
        Jawaban: [A/B/C/D]

        **Soal 2:**
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
    
    def generate_mcq(self, question: str, context: str):
        num_questions = 10  # Default
        num_match = re.search(r'(\d+)\s*(?:soal|pertanyaan|question)', question, re.IGNORECASE)
        if num_match:
            num_questions = int(num_match.group(1))
            print(f"Detected request for {num_questions} questions")
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user', 
                    'content': self.format_mcq_prompt(question, context, num_questions)
                }]
            )
            
            content = response['message']['content']
            
            print(f"Response sample (first 300 chars):\n{content[:300]}")
            
            parsed_json = parse_mcq_text(content)
            
            if parsed_json["total_questions"] < num_questions:
                print(f"Only parsed {parsed_json['total_questions']} of {num_questions} questions.")
                print("Attempting to reformat and parse again...")
                
                enhanced_content = self.enhance_content_format(content)
                parsed_json = parse_mcq_text(enhanced_content)
                
                if parsed_json["total_questions"] < num_questions:
                    print("Still missing questions. Making another API call with stricter format...")
                    second_prompt = self.format_mcq_prompt_strict(question, context, num_questions, 
                                                                parsed_json["total_questions"])
                    
                    second_response = ollama.chat(
                        model=self.model,
                        messages=[{
                            'role': 'user', 
                            'content': second_prompt
                        }]
                    )
                    
                    second_content = second_response['message']['content']
                    second_parsed = parse_mcq_text(second_content)
                    
                    all_questions = parsed_json["questions"] + second_parsed["questions"]
                    
                    unique_questions = {}
                    for q in all_questions:
                        if q["number"] not in unique_questions:
                            unique_questions[q["number"]] = q
                    
                    combined_questions = list(unique_questions.values())
                    combined_questions.sort(key=lambda x: x["number"])
                    
                    parsed_json = {
                        "total_questions": len(combined_questions),
                        "questions": combined_questions
                    }
            
            return parsed_json
            
        except Exception as e:
            import traceback
            print(f"Error in generate_mcq: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error processing LLM response: {str(e)}")
    
    def format_mcq_prompt_strict(self, question: str, context: str, num_questions: int, 
                                parsed_questions: int) -> str:
        start_from = parsed_questions + 1
        remaining = num_questions - parsed_questions
        
        return f"""Anda adalah seorang dosen berpengalaman. 
        Sebelumnya Anda telah membuat {parsed_questions} soal pilihan ganda, 
        tetapi masih perlu membuat {remaining} soal lagi (dari Soal {start_from} hingga Soal {num_questions}).
        
        **IKUTI FORMAT INI DENGAN TEPAT untuk setiap soal:**
        
        **Soal [nomor]:**
        [Pertanyaan lengkap]
        A) [Pilihan A]
        B) [Pilihan B]
        C) [Pilihan C]
        D) [Pilihan D]
        Jawaban: [A/B/C/D]
        
        Buat soal mulai dari Soal {start_from} sampai Soal {num_questions}.
        
        **SANGAT PENTING:**
        - Pastikan ada baris "Jawaban: X" setelah setiap pilihan D
        - Pilihan jawaban harus A, B, C, atau D
        - Format harus persis seperti di atas
        - Jangan ada teks tambahan sebelum "Soal [nomor]:"
        - Jangan ada teks tambahan setelah "Jawaban: [A/B/C/D]"
        
        **Konteks:** {context}
        **Permintaan:** {question}
        """
    
    @staticmethod
    def enhance_content_format(content: str) -> str:
        lines = content.split('\n')
        enhanced_lines = []
        
        option_count = 0
        current_options = {}
        current_question = None
        
        for line in lines:
            stripped = line.strip()
            
            question_match = re.search(r'(?:\*\*)?Soal\s+(\d+)(?:\*\*)?[:\.]?', stripped)
            if question_match:
                if current_question and option_count == 4 and len(current_options) == 4:
                    if not any("jawaban" in l.lower() for l in enhanced_lines[-5:]):
                        enhanced_lines.append(f"Jawaban: A")
                
                current_question = question_match.group(1)
                option_count = 0
                current_options = {}
                enhanced_lines.append(line)
                continue
            
            option_match = re.match(r'^([A-D])[\s\)\.:]+\s*(.*)', stripped)
            if option_match:
                option_letter = option_match.group(1)
                current_options[option_letter] = option_match.group(2)
                option_count += 1
                enhanced_lines.append(line)
                continue
            
            if "jawaban" in stripped.lower() and ":" in stripped:
                enhanced_lines.append(line)
                continue
            
            if option_count == 4 and len(current_options) == 4 and not any("jawaban" in l.lower() for l in enhanced_lines[-4:]):
                enhanced_lines.append(line)
                
                if not re.search(r'[jJ]awaban', stripped):
                    enhanced_lines.append(f"Jawaban: A")
                continue
            
            enhanced_lines.append(line)
        
        if option_count == 4 and len(current_options) == 4:
            if not any("jawaban" in l.lower() for l in enhanced_lines[-5:]):
                enhanced_lines.append(f"Jawaban: A")
        
        return '\n'.join(enhanced_lines)
    
    def generate_json_response(self, question: str, context: str):
        pass
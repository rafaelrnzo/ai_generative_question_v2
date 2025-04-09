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

            **PERHATIAN PENTING: JANGAN MEMBUAT SOAL PILIHAN GANDA. SAYA HANYA MEMBUTUHKAN SOAL ESSAY TANPA OPTION A, B, C, D.**

            **Instruksi WAJIB untuk format output:**  
            - **WAJIB membuat soal dalam format ESSAY dengan pertanyaan terbuka.**
            - **DILARANG KERAS membuat soal pilihan ganda atau multiple choice.**
            - **DILARANG membuat opsi jawaban dengan format A), B), C), D) atau sejenisnya.**  
            - **Setiap soal harus berupa pertanyaan yang membutuhkan jawaban panjang dan penjelasan.**  
            - **Setiap soal harus memiliki jawaban yang lengkap dan komprehensif.**
            - **Tidak boleh menggunakan huruf A, B, C, atau D dalam soal maupun jawaban.**  
            - **WAJIB menggunakan format berikut:**  

            **FORMAT OUTPUT:**
            Soal 1:
            [Tulis pertanyaan ESSAY di sini. Pastikan pertanyaannya kompleks dan membutuhkan jawaban panjang]
            Jawaban: [Tulis jawaban ESSAY yang lengkap dan mendalam di sini]

            Soal 2:
            [Tulis pertanyaan ESSAY di sini. Pastikan pertanyaannya kompleks dan membutuhkan jawaban panjang]
            Jawaban: [Tulis jawaban ESSAY yang lengkap dan mendalam di sini]

            **CONTOH SOAL ESSAY YANG BENAR:**
            Soal 1: 
            Jelaskan prinsip kerja sistem bahan bakar pada kapal dan bagaimana komponen-komponennya saling berinteraksi untuk mengoptimalkan efisiensi bahan bakar?
            Jawaban: Sistem bahan bakar pada kapal terdiri dari beberapa komponen utama yang bekerja secara terintegrasi. Dimulai dari tangki penyimpanan, bahan bakar akan melalui proses pemurnian menggunakan separator dan purifier untuk menghilangkan kontaminan. Kemudian, bahan bakar dipompa ke tangki harian (service tank) sebelum masuk ke sistem injeksi. Dalam proses ini, tekanan dan temperatur bahan bakar harus dijaga pada nilai optimal untuk memastikan atomisasi sempurna saat injeksi ke dalam silinder. Interaksi antar komponen diatur oleh sistem kontrol yang memastikan timing yang tepat antara suplai bahan bakar dan kebutuhan mesin. Efisiensi sistem bergantung pada pemeliharaan filter, kalibrasi pompa, dan keselarasan operasional seluruh komponen.

            **CONTOH SOAL YANG SALAH (JANGAN BUAT SEPERTI INI):**
            Soal 1:
            Apa fungsi utama dari MPK dalam sistem bahan bakar kapal?
            A) Mengatur aliran bahan bakar dari tangki harian
            B) Menjaga keseimbangan dan kestabilan sistem bahan bakar
            C) Mengontrol pengeluaran bahan bakar
            D) Membuat komponen lain dalam sistem bekerja optimal

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
            # First attempt
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user', 
                    'content': self.format_essay_prompt(question, context, num_questions)
                }]
            )
            
            content = response['message']['content']
            print(f"LLM Response (first 300 chars):\n{content[:300]}")  # Debugging

            # Check if response contains A), B), C), D) options
            if re.search(r'[A-D]\)', content) or re.search(r'[A-D]\s*\)', content):
                print("Warning: Detected multiple-choice format in response. Retrying with stronger constraints...")
                # Second attempt with even stronger constraint
                emphasized_prompt = f"""PERHATIAN: Respons sebelumnya masih berisi pilihan ganda dengan opsi A), B), C), D).
                
                SAYA SANGAT MENEKANKAN: JANGAN MEMBUAT SOAL PILIHAN GANDA. SAYA HANYA MEMBUTUHKAN SOAL ESSAY.

                Buatlah {num_questions} soal essay tentang {context} tanpa opsi pilihan ganda.
                Setiap soal harus berupa pertanyaan terbuka yang membutuhkan jawaban penjelasan panjang.
                Format yang benar adalah:

                Soal 1: [Pertanyaan essay]
                Jawaban: [Jawaban lengkap]

                Soal 2: [Pertanyaan essay]
                Jawaban: [Jawaban lengkap]

                Jangan gunakan format A), B), C), D) atau sejenisnya.
                Pastikan soal merupakan pertanyaan essay yang kompleks sesuai dengan konteks: {context}
                """
                
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {'role': 'user', 'content': self.format_essay_prompt(question, context, num_questions)},
                        {'role': 'assistant', 'content': content},
                        {'role': 'user', 'content': emphasized_prompt}
                    ]
                )
                content = response['message']['content']
                print(f"Second attempt response (first 300 chars):\n{content[:300]}")

            # Parse the response into structured format
            parsed_json = self.parse_essay_text(content, num_questions)

            # If we still don't have enough questions, try one more time
            if parsed_json["total_questions"] < num_questions:
                print("Warning: Incomplete questions extracted. Retrying with a modified prompt...")
                new_prompt = f"""Saya perlu tepat {num_questions} soal essay tentang {context}.
                
                Jawaban Anda sebelumnya hanya menghasilkan {parsed_json["total_questions"]} soal yang dapat diidentifikasi.
                
                Mohon berikan {num_questions} soal essay dengan format yang jelas:
                
                Soal 1: [Pertanyaan essay]
                Jawaban: [Jawaban lengkap]
                
                Soal 2: [Pertanyaan essay]
                Jawaban: [Jawaban lengkap]
                
                Dan seterusnya hingga Soal {num_questions}.
                """
                
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {'role': 'user', 'content': self.format_essay_prompt(question, context, num_questions)},
                        {'role': 'assistant', 'content': content},
                        {'role': 'user', 'content': new_prompt}
                    ]
                )
                content = response['message']['content']
                parsed_json = self.parse_essay_text(content, num_questions)

            # Post-process to remove any remaining multiple-choice formatting
            self.clean_multiple_choice_format(parsed_json)
            
            return parsed_json
            
        except Exception as e:
            import traceback
            print(f"Error in generate_essay: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error processing LLM response: {str(e)}")
    
    @staticmethod
    def clean_multiple_choice_format(parsed_json):
        """Remove any remaining multiple-choice formatting from questions and answers"""
        for q in parsed_json["questions"]:
            # Remove any options like A), B), C), D) from both questions and answers
            q["question"] = re.sub(r'\n[A-D]\)[^\n]+', '', q["question"])
            q["answer"] = re.sub(r'\n[A-D]\)[^\n]+', '', q["answer"])
            
            # If the question still looks like a multiple-choice stem, expand it
            if len(q["question"].strip()) < 100 and not q["question"].strip().endswith('?'):
                q["question"] = f"Jelaskan secara detail tentang {q['question'].strip()}?"
    
    @staticmethod
    def parse_essay_text(content: str, expected_count=10):
        questions = []
        
        # More robust regex pattern to extract questions and answers
        pattern = r'(?:Soal\s*(\d+):?|(?<!\w)(\d+)\.)\s*(.*?)(?:\n+(?:Jawaban:?|Jawaban\s*\d+:?)\s*(.*?)(?=\n+(?:Soal\s*\d+:|(?<!\w)\d+\.)|$))'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if not matches:
            # Fallback pattern if the first one doesn't work
            question_pattern = r'(?:^|\n)(?:Soal\s*\d+:?|(?<!\w)\d+\.)?\s*(.*?)(?=\n+(?:Jawaban:?|Jawaban\s*\d+:?))'
            answer_pattern = r'(?:Jawaban:?|Jawaban\s*\d+:?)\s*(.*?)(?=\n+(?:Soal\s*\d+:|(?<!\w)\d+\.)|$)'
            
            question_matches = re.findall(question_pattern, content, re.DOTALL)
            answer_matches = re.findall(answer_pattern, content, re.DOTALL)
            
            for i in range(min(len(question_matches), len(answer_matches))):
                questions.append({
                    "number": i + 1,
                    "question": question_matches[i].strip(),
                    "answer": answer_matches[i].strip()
                })
        else:
            for i, match in enumerate(matches):
                # The match format might be (num, '', question, answer) or ('', num, question, answer)
                num = match[0] if match[0] else match[1]
                question = match[2].strip()
                answer = match[3].strip()
                
                questions.append({
                    "number": i + 1,  # Use sequential numbering for consistency
                    "question": question,
                    "answer": answer
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
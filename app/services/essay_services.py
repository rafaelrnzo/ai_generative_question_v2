import json
import re
import ollama
from fastapi import HTTPException
from core.config import OLLAMA_HOST, OLLAMA_MODEL

class EssayService:
    def __init__(self, llm_service=None):
        """
        Initialize the EssayService with an optional LLMService instance.
        If not provided, it will configure Ollama directly.
        
        Args:
            llm_service: An instance of LLMService. If None, direct Ollama configuration is used.
        """
        self.llm_service = llm_service
        
        if not llm_service:
            ollama.base_url = OLLAMA_HOST
            self.model = OLLAMA_MODEL
        else:
            self.model = llm_service.model
    
    def format_essay_prompt(self, question: str, context: str, num_essays: int, language: str) -> str:
        """
        Format the prompt for essay generation based on the selected language.
        
        Args:
            question: The user's request
            context: The context information to base essays on
            num_essays: Number of essays to generate
            language: "english" or "indonesian"
            
        Returns:
            Formatted prompt string
        """
        if language.lower() == "indonesian":
            return f"""Anda adalah dosen berpengalaman yang ahli dalam pembuatan soal essay berkualitas tinggi.

**Tugas:**
Buatlah {num_essays} soal essay yang menguji pemahaman mendalam tentang topik berdasarkan konteks yang diberikan.

**Instruksi:**
- Buat sebanyak {num_essays} soal essay dengan jawaban lengkap.
- Setiap soal harus menguji pemahaman konseptual, analisis kritis, atau kemampuan aplikasi.
- Berikan jawaban model yang komprehensif untuk setiap soal.
- Soal harus mencakup berbagai tingkat kesulitan kognisi (mengingat, memahami, menganalisis, mengevaluasi).

**Format output yang HARUS diikuti:**
```json
{{
  "total_questions": {num_essays},
  "questions": [
    {{
      "id": 1,
      "question": "Tuliskan pertanyaan essay pertama di sini...",
      "answer": "Jawaban komprehensif untuk pertanyaan pertama..."
    }},
    {{
      "id": 2,
      "question": "Tuliskan pertanyaan essay kedua di sini...",
      "answer": "Jawaban komprehensif untuk pertanyaan kedua..."
    }},
    ...
  ]
}}
```

**PENTING:**
- Output HARUS berupa JSON yang valid dengan format persis seperti di atas.
- Setiap soal harus memiliki ID unik, teks pertanyaan, dan teks jawaban.
- Jawaban harus komprehensif dan menyeluruh.

**Konteks:** {context}

**Permintaan:** {question}
"""
        else:  # English
            return f"""You are an experienced professor who specializes in creating high-quality essay questions.

**Task:**
Create {num_essays} essay questions that test deep understanding of the topic based on the provided context.

**Instructions:**
- Create exactly {num_essays} essay questions with complete answers.
- Each question should test conceptual understanding, critical analysis, or application ability.
- Provide a comprehensive model answer for each question.
- Questions should cover various levels of cognitive difficulty (recall, comprehension, analysis, evaluation).

**Output format that MUST be followed:**
```json
{{
  "total_questions": {num_essays},
  "questions": [
    {{
      "id": 1,
      "question": "Write the first essay question here...",
      "answer": "Comprehensive answer for the first question..."
    }},
    {{
      "id": 2,
      "question": "Write the second essay question here...",
      "answer": "Comprehensive answer for the second question..."
    }},
    ...
  ]
}}
```

**IMPORTANT:**
- Output MUST be valid JSON with format exactly as shown above.
- Each question must have a unique ID, question text, and answer text.
- Answers should be comprehensive and thorough.

**Context:** {context}

**Request:** {question}
"""

    def generate_essays(self, question: str, context: str, num_essays: int = 1, language: str = "english") -> dict:
        """
        Generate essay questions and answers based on the provided context.
        
        Args:
            question: The user's request for essay generation
            context: The context information
            num_essays: Number of essay questions to generate
            language: The language to generate essays in ("english" or "indonesian")
            
        Returns:
            A dictionary containing the generated essays
        """
        # Normalize language
        language = language.lower().strip()
        if language not in ["english", "indonesian"]:
            language = "english"  # Default to English if unrecognized
        
        # Ensure num_essays is a valid number
        try:
            num_essays = int(num_essays)
            if num_essays < 1:
                num_essays = 1
            elif num_essays > 10:  # Limit to reasonable number
                num_essays = 10
        except (ValueError, TypeError):
            num_essays = 1
        
        try:
            # Format the prompt
            prompt = self.format_essay_prompt(question, context, num_essays, language)
            
            # Call Ollama directly or through the llm_service
            if self.llm_service:
                # Assuming llm_service has a method to call the LLM
                raw_response = self.llm_service.call_llm(prompt)
            else:
                # Call Ollama directly
                response = ollama.chat(
                    model=self.model,
                    messages=[{
                        'role': 'user', 
                        'content': prompt
                    }]
                )
                raw_response = response['message']['content']
            
            # Extract JSON from the response
            return self.parse_essay_response(raw_response, num_essays)
            
        except Exception as e:
            import traceback
            print(f"Error in generate_essays: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error processing LLM response: {str(e)}")
    
    def parse_essay_response(self, content: str, expected_count: int) -> dict:
        """
        Parse the LLM's response and extract valid JSON.
        
        Args:
            content: The raw text response from the LLM
            expected_count: Expected number of questions
            
        Returns:
            A dictionary containing the parsed response
        """
        # Default response if parsing fails
        default_response = {
            "total_questions": 0,
            "questions": []
        }
        
        try:
            # Look for JSON in the response
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            
            if not json_match:
                # Try different pattern
                json_match = re.search(r'{[\s\S]*"total_questions"[\s\S]*"questions"[\s\S]*}', content)
            
            if json_match:
                json_content = json_match.group(1)
            else:
                # If no clear JSON block, take the whole content
                json_content = content
            
            # Clean up the content to ensure it's valid JSON
            # Remove any non-JSON leading/trailing content
            json_content = re.sub(r'^[^{]*', '', json_content)
            json_content = re.sub(r'[^}]*$', '', json_content)
            
            # Parse the JSON
            parsed_data = json.loads(json_content)
            
            # Validate the structure
            if "total_questions" not in parsed_data or "questions" not in parsed_data:
                print("Missing required fields in parsed JSON")
                return default_response
            
            # Ensure correct question IDs and update total_questions
            questions = parsed_data["questions"]
            for i, question in enumerate(questions):
                # Ensure each question has id, question, and answer
                if "id" not in question:
                    question["id"] = i + 1
                if "question" not in question or "answer" not in question:
                    # Skip incomplete questions
                    questions.pop(i)
            
            # Update total questions count
            parsed_data["total_questions"] = len(questions)
            
            # If no questions were parsed, return default
            if parsed_data["total_questions"] == 0:
                print("No valid questions found in the response")
                return default_response
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Raw content: {content[:500]}...")
            
            # Attempt to extract individual questions if JSON parsing failed
            questions = []
            question_matches = re.finditer(r'id["\s:]+(\d+)["\s,]+question["\s:]+([^"]+)["\s,]+answer["\s:]+([^"]+)', 
                                          content, re.DOTALL)
            
            for i, match in enumerate(question_matches):
                if i >= expected_count:
                    break
                    
                questions.append({
                    "id": int(match.group(1)) if match.group(1).isdigit() else i + 1,
                    "question": match.group(2).strip(),
                    "answer": match.group(3).strip()
                })
            
            return {
                "total_questions": len(questions),
                "questions": questions
            }
            
        except Exception as e:
            print(f"Error parsing essay response: {str(e)}")
            return default_response
            
    def call_llm(self, prompt):
        """
        Direct method to call the LLM if no llm_service is provided.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The raw text response from the LLM
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user', 
                    'content': prompt
                }]
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error calling LLM: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error calling LLM: {str(e)}")
import re 
from fastapi import HTTPException

def parse_mcq_text(content: str):
    try:
        questions = []
        
        # Look for both Indonesian and English question patterns
        question_markers = re.finditer(r'(?:\*\*)?(?:Soal|Question)\s+(\d+)(?:\*\*)?[:\.]?', content)
        
        positions = [(int(match.group(1)), match.start()) for match in question_markers]
        positions.sort(key=lambda x: x[0])  # Sort by question number
        
        if not positions:
            print("No standard question markers found. Trying alternative patterns...")
            alternative_markers = re.finditer(r'(?:^|\n)(?:\*\*)?(\d+)(?:\*\*)?[:\.]', content)
            positions = [(int(match.group(1)), match.start()) for match in alternative_markers]
            positions.sort(key=lambda x: x[0])
        
        if positions:
            for i in range(len(positions)):
                start_pos = positions[i][1]
                end_pos = positions[i+1][1] if i < len(positions) - 1 else len(content)
                question_content = content[start_pos:end_pos].strip()
                
                try:
                    question_data = parse_single_question(question_content, positions[i][0])
                    if question_data:
                        questions.append(question_data)
                except Exception as e:
                    print(f"Error parsing question at position {start_pos}: {str(e)}")
        else:
            # Try both Indonesian and English patterns for fallback
            raw_questions = re.split(r'(?:\*\*)?(?:Soal|Question)(?:\*\*)?:', content)
            if raw_questions and not raw_questions[0].strip():
                raw_questions = raw_questions[1:]
            
            for i, raw_question in enumerate(raw_questions, 1):
                try:
                    question_data = parse_single_question(raw_question, i)
                    if question_data:
                        questions.append(question_data)
                except Exception as e:
                    print(f"Error in fallback parsing of question {i}: {str(e)}")
        
        questions.sort(key=lambda q: q['number'])
        
        for i, q in enumerate(questions, 1):
            q['number'] = i
            
        return {
            "total_questions": len(questions),
            "questions": questions
        }
    except Exception as e:
        import traceback
        print(f"Error parsing MCQ text: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error parsing MCQ response: {str(e)}")
    
def parse_single_question(question_content, question_number):
    # Remove question marker at the beginning
    question_content = re.sub(r'^(?:\*\*)?(?:Soal|Question)\s+\d+(?:\*\*)?[:\.]?\s*', '', question_content)
    
    lines = question_content.split('\n')
    clean_lines = [line.strip() for line in lines if line.strip()]
    
    question_text = ""
    option_start_idx = -1
    
    # Find where the options start
    for i, line in enumerate(clean_lines):
        if re.match(r'^[A-D][\)\.]', line) or re.match(r'^[A-D]\s*[\)\.]', line):
            option_start_idx = i
            break
        question_text += line + " "
    
    question_text = question_text.strip()
    
    if option_start_idx == -1:
        for i, line in enumerate(clean_lines):
            if re.search(r'[A-D][\)\.].*', line):
                option_start_idx = i
                break
    
    options = {'A': '', 'B': '', 'C': '', 'D': ''}
    answer = None
    
    if option_start_idx != -1:
        # Process options and look for answer
        option_lines = clean_lines[option_start_idx:]
        current_option = None
        
        for i, line in enumerate(option_lines):
            # Check for option markers
            option_match = re.match(r'^([A-D])[\s\)\.:]+\s*(.*)', line)
            if option_match:
                current_option = option_match.group(1)
                options[current_option] = option_match.group(2).strip()
                continue
                
            # Check for answer markers with multiple patterns
            answer_patterns = [
                r'(?:[Jj]awaban|[Aa]nswer)[\s\:\=]+([A-D])',  # Standard format
                r'[Cc]orrect\s+[Aa]nswer[\s\:\=]+([A-D])',    # "Correct Answer: X" format
                r'[Cc]orrect[\s\:\=]+([A-D])',                # "Correct: X" format
                r'[Tt]he\s+[Aa]nswer\s+is[\s\:\=]+([A-D])',   # "The answer is X" format
                r'[Kk]ey[\s\:\=]+([A-D])'                     # "Key: X" format
            ]
            
            for pattern in answer_patterns:
                answer_match = re.search(pattern, line)
                if answer_match:
                    answer = answer_match.group(1)
                    break
            
            # If this line is a continuation of the previous option
            if current_option and not option_match and not answer:
                options[current_option] += " " + line
    
    # If answer wasn't found in the options section, check the entire content
    if not answer:
        # Try all the answer patterns on the full content
        answer_patterns = [
            r'(?:[Jj]awaban|[Aa]nswer)[\s\:\=]+([A-D])',
            r'[Cc]orrect\s+[Aa]nswer[\s\:\=]+([A-D])',
            r'[Cc]orrect[\s\:\=]+([A-D])',
            r'[Tt]he\s+[Aa]nswer\s+is[\s\:\=]+([A-D])',
            r'[Kk]ey[\s\:\=]+([A-D])'
        ]
        
        for pattern in answer_patterns:
            answer_match = re.search(pattern, question_content)
            if answer_match:
                answer = answer_match.group(1)
                break
    
    # Look for bold formatting that might indicate the answer
    if not answer:
        bold_match = re.search(r'\*\*\s*([A-D])\s*\*\*', question_content)
        if bold_match:
            answer = bold_match.group(1)
    
    # Check for answer indicators like "(correct)" next to an option
    if not answer:
        for option_letter in options:
            if re.search(r'(?:\(correct\)|\(right\)|\(true\))', options[option_letter], re.IGNORECASE):
                answer = option_letter
                # Clean up the option text
                options[option_letter] = re.sub(r'\s*(?:\(correct\)|\(right\)|\(true\))\s*', '', options[option_letter], flags=re.IGNORECASE)
                break
    
    # Default to A only if we have all options and couldn't find an answer
    if not answer and all(options.values()):
        print(f"Warning: No answer found for question {question_number}. Using default answer A.")
        answer = 'A'
    
    has_all_options = all(options.values())
    print(f"Question {question_number} parsing:")
    print(f"  Question text: {len(question_text) > 0}")
    print(f"  Has all options: {has_all_options}")
    print(f"  Options: {options}")
    print(f"  Answer found: {answer is not None}")
    
    if question_text and has_all_options and answer:
        return {
            "number": question_number,
            "question": question_text.strip(),
            "options": options,
            "answer": answer
        }
    else:
        print(f"Missing data for question {question_number}:")
        print(f"  Question text: {bool(question_text)}")
        print(f"  Options: {options}")
        print(f"  Answer: {answer}")
        print(f"  Raw content: {question_content[:100]}...")
        return None
    
    # You are an **experienced lecturer in the field of {context}**. 
    #         Your task is to create **{num_questions} high-quality multiple-choice questions**.

    #         **CRITICAL FORMATTING INSTRUCTIONS:**
    #         - Each question must have exactly four options labeled A, B, C, and D.
    #         - Each question MUST end with "Answer: X" where X is one of A, B, C, or D.
    #         - DO NOT include any additional text, explanations, or notes - ONLY the questions, options, and answers.
    #         - Follow EXACTLY the format of the example below.
    #         - Must contain the option of the answer (A, B, C, D)
            
    #         **REQUIRED FORMAT - FOLLOW THIS EXACTLY:**
    #         always follow this example layout
    #         this is the example:
    #         What is the primary display technology used in the Samsung Galaxy S25?

    #         A) OLED
    #         B) Dynamic LTPO AMOLED 2X
    #         C) Super AMOLED
    #         D) Quantum Dot

    #         Answer: B) Dynamic LTPO AMOLED 2X <- this is mandatory
            

    #         **IMPORTANT:** 
    #         - The answer MUST be on its own line immediately after option D
    #         - The answer MUST be in the format "Answer: X" where X is A, B, C, or D
    #         - Use only one letter for the answer (A, B, C, or D)
    #         - Do not include any explanations or additional information beyond the question, options, and answer

    #         **Context:** {context}
    #         **Task:** {question}
    #         **Number of questions to generate: {num_questions}**
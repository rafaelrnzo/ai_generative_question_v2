import re 
from fastapi import HTTPException

def parse_mcq_text(content: str):
    try:
        questions = []
        
        # First detect if we have the common pattern with "Soal X" format
        # Look for patterns like "Soal 1", "**Soal 1**", etc.
        question_markers = re.finditer(r'(?:\*\*)?Soal\s+(\d+)(?:\*\*)?[:\.]?', content)
        
        # Get all positions where questions start
        positions = [(int(match.group(1)), match.start()) for match in question_markers]
        positions.sort(key=lambda x: x[0])  # Sort by question number
        
        if not positions:
            # If no markers found, try other patterns
            print("No standard question markers found. Trying alternative patterns...")
            # Try to find numbered patterns at the start of lines
            alternative_markers = re.finditer(r'(?:^|\n)(?:\*\*)?(\d+)(?:\*\*)?[:\.]', content)
            positions = [(int(match.group(1)), match.start()) for match in alternative_markers]
            positions.sort(key=lambda x: x[0])
        
        # If we found question markers, extract questions using positions
        if positions:
            for i in range(len(positions)):
                start_pos = positions[i][1]
                end_pos = positions[i+1][1] if i < len(positions) - 1 else len(content)
                question_content = content[start_pos:end_pos].strip()
                
                try:
                    # Parse this individual question
                    question_data = parse_single_question(question_content, positions[i][0])
                    if question_data:
                        questions.append(question_data)
                except Exception as e:
                    print(f"Error parsing question at position {start_pos}: {str(e)}")
        else:
            # Fallback: split by "Soal" without numbers
            raw_questions = re.split(r'(?:\*\*)?Soal(?:\*\*)?:', content)
            if raw_questions and not raw_questions[0].strip():
                raw_questions = raw_questions[1:]
            
            for i, raw_question in enumerate(raw_questions, 1):
                try:
                    question_data = parse_single_question(raw_question, i)
                    if question_data:
                        questions.append(question_data)
                except Exception as e:
                    print(f"Error in fallback parsing of question {i}: {str(e)}")
        
        # Sort questions by their number
        questions.sort(key=lambda q: q['number'])
        
        # Renumber questions consecutively
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
    """Parse a single question from the content"""
    # Strip any question numbers from the beginning
    question_content = re.sub(r'^(?:\*\*)?(?:Soal\s+)?\d+(?:\*\*)?[:\.]?\s*', '', question_content)
    
    lines = question_content.split('\n')
    clean_lines = [line.strip() for line in lines if line.strip()]
    
    # Extract question text
    question_text = ""
    option_start_idx = -1
    
    for i, line in enumerate(clean_lines):
        # Stop when we hit an option line
        if re.match(r'^[A-D][\)\.]', line) or re.match(r'^[A-D]\s*[\)\.]', line):
            option_start_idx = i
            break
        question_text += line + " "
    
    question_text = question_text.strip()
    
    # If we couldn't find option markers, try another approach
    if option_start_idx == -1:
        # Look for lines with option-like patterns
        for i, line in enumerate(clean_lines):
            if re.search(r'[A-D][\)\.].*', line):
                option_start_idx = i
                break
    
    # Extract options
    options = {'A': '', 'B': '', 'C': '', 'D': ''}
    answer = None
    
    if option_start_idx != -1:
        # Process options
        for i in range(option_start_idx, len(clean_lines)):
            # Look for option markers like A), B), etc.
            option_match = re.match(r'^([A-D])[\s\)\.:]+\s*(.*)', clean_lines[i])
            if option_match:
                option_letter = option_match.group(1)
                option_text = option_match.group(2).strip()
                options[option_letter] = option_text
            
            # Look for answer markers in this line
            answer_match = re.search(r'[Jj]awaban[\s\:\=]+([A-D])', clean_lines[i])
            if answer_match:
                answer = answer_match.group(1)
    
    # If we still don't have an answer, search the entire content
    if not answer:
        answer_match = re.search(r'[Jj]awaban[\s\:\=]+([A-D])', question_content)
        if answer_match:
            answer = answer_match.group(1)
    
    # If we still don't have an answer, try to infer it from formatting
    if not answer:
        # Look for bold text that might be the answer
        bold_match = re.search(r'\*\*\s*([A-D])\s*\*\*', question_content)
        if bold_match:
            answer = bold_match.group(1)
    
    # As a last resort, if all options are filled but no answer, take the first option
    if not answer and all(options.values()):
        print(f"Warning: No answer found for question {question_number}. Using default answer A.")
        answer = 'A'  # Default to A if we can't find the answer
    
    # Debug output
    has_all_options = all(options.values())
    print(f"Question {question_number} parsing:")
    print(f"  Question text: {len(question_text) > 0}")
    print(f"  Has all options: {has_all_options}")
    print(f"  Options: {options}")
    print(f"  Answer found: {answer is not None}")
    
    # Only add questions that have all the necessary components
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
        print(f"  Raw text: {question_content[:50]}...")
        return None
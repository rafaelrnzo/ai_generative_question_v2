def is_mcq_request(question_text):
    mcq_keywords = ['soal', 'pilihan ganda', 'mcq', 'multiple choice', 'pertanyaan', 'questions', 'question']
    return any(keyword in question_text.lower() for keyword in mcq_keywords)


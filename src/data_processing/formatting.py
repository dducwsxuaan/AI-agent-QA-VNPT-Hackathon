import re

def format_choices(choices_list: list) -> str:
    """Chuyển list ['a', 'b'] thành string 'A. a \\n B. b'"""
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    formatted = []
    for i, choice in enumerate(choices_list):
        if i < len(labels):
            formatted.append(f"{labels[i]}. {choice}")
    return "\n".join(formatted)

def extract_answer_key(text_response: str) -> str:
    """
    Trích xuất ký tự đáp án (A, B, C, D) từ câu trả lời của LLM.
    Mặc định trả về "A" nếu không tìm thấy.
    """
    # Pattern tìm kiếm: "Answer: A" hoặc "Đáp án: B"
    pattern = r"(?:Đáp án|Answer|Result)[:\s]+([A-Z])"
    match = re.search(pattern, text_response, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).upper()
    
    # Fallback: Nếu câu trả lời chỉ có 1 ký tự A-D ở đầu dòng
    match_fallback = re.search(r"^([A-D])\.?", text_response.strip())
    if match_fallback:
        return match_fallback.group(1).upper()
        
    return "A" # Default an toàn 

"""Answer extraction and validation utilities.

Consolidates answer-related logic:
- Extraction from LLM responses (CoT format)
- Validation against valid choices
- Normalization with fallback defaults
"""

import re
import string

from src.utils.logging import print_log


def extract_answer(response: str) -> str:
    """
    Trích xuất đáp án (A, B, C, D...) từ phản hồi của LLM.
    Ưu tiên cấu trúc: "Answer: X" hoặc "Đáp án: X".
    """
    if not response:
        return "A"
        
    clean_response = response.strip()
    
    # 1. ƯU TIÊN CAO NHẤT: Tìm pattern "Answer: A" hoặc "Đáp án: B"
    # Regex giải thích:
    # - (?:Answer|Đáp án|Lựa chọn): Tìm từ khóa
    # - .*?: Chấp nhận bất kỳ ký tự nào ở giữa (ví dụ "Answer is")
    # - [:punct:\s]*: Chấp nhận dấu hai chấm, dấu sao markdown (**), khoảng trắng
    # - ([A-Z]): Bắt ký tự in hoa (Group 1)
    # - (?= ...): Lookahead - Kiểm tra ký tự ngay sau nó (để tránh bắt nhầm chữ cái đầu của từ, ví dụ "Answer: About")
    #       [\s.)]: Phải là khoảng trắng, dấu chấm, hoặc dấu đóng ngoặc
    #       |$: Hoặc là kết thúc chuỗi
    
    match_candidates = re.findall(
        r"(?:Answer|Đáp án|Lựa chọn|Ans|Kết quả|Chốt).*?[:\s*#]+([A-Z])(?=[\s.)]|$)", 
        clean_response, 
        re.IGNORECASE | re.DOTALL # DOTALL quan trọng để .*? băng qua được xuống dòng
    )
    
    bullet_candidates = re.findall(
        r"(?:^|\n)[\s*#]*([A-Z])[.)](?=\s|$)", 
        clean_response
    )
    if bullet_candidates:
        return bullet_candidates[-1].upper()

    # --- CHIẾN THUẬT 3: Fallback cuối cùng (Tìm chữ cái in hoa đứng lẻ) ---
    # Chỉ tìm A, B, C, D, E, F để tránh bắt nhầm các chữ cái khác (như T trong 'Thân ái', H trong 'Hết')
    # Logic: Ký tự in hoa đứng độc lập
    
    standalone_candidates = re.findall(
        r"(?<!\w)([A-F])(?!\w)", 
        clean_response
    )
    
    if standalone_candidates:
        return standalone_candidates[-1].upper()

    return "A"


def validate_answer(answer: str, num_choices: int) -> tuple[bool, str]:
    """Validate if answer is within valid range and normalize it."""
    valid_answers = string.ascii_uppercase[:num_choices]
    
    if answer.upper() in valid_answers:
        return True, answer.upper()
    
    return False, answer


def normalize_answer(
    answer: str,
    num_choices: int,
    question_id: str | None = None,
    default: str = "A",
) -> str:
    """Normalize and validate answer with fallback to default."""
    is_valid, normalized = validate_answer(answer, num_choices)
    
    if not is_valid:
        if question_id:
            print_log(
                f"        [Warning] Invalid answer '{answer}' for {question_id}, "
                f"defaulting to {default}"
            )
        return default
    
    return normalized


def extract_and_normalize(
    response: str,
    num_choices: int,
    question_id: str | None = None,
    default: str = "A",
) -> str:
    """Extract answer from response and normalize it."""
    extracted = extract_answer(response, max_choices=num_choices)
    return normalize_answer(extracted, num_choices, question_id, default)
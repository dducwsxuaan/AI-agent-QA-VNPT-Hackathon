"""Answer extraction and validation utilities.

Consolidates answer-related logic:
- Extraction from LLM responses (CoT format)
- Validation against valid choices
- Normalization with fallback defaults
"""

import re
import string

from src.utils.logging import print_log


def extract_answer(response: str, max_choices: int = 15) -> str:
    """
    Trích xuất đáp án từ phản hồi LLM.
    Mặc định max_choices=15 (A-O) để cân bằng giữa độ bao phủ và an toàn.
    """
    if not response:
        return "A"
        
    clean_response = response.strip()
    
    # Tạo tập nhãn hợp lệ: {'A', 'B', ..., 'O'}
    valid_labels = set(string.ascii_uppercase[:max_choices])

    # --- TẦNG 1: Ưu tiên cao nhất (Answer: I) ---
    # Cấu trúc này rất an toàn, kể cả với chữ I
    explicit_matches = re.findall(
        r"(?:Answer|Đáp án|Lựa chọn|Ans|Kết quả|Chốt).*?[:：\s]+(?:[*#\"'\s]*)([A-Z])(?=[\s.)]|$)", 
        clean_response, 
        re.IGNORECASE | re.DOTALL
    )
    
    if explicit_matches:
        final_match = explicit_matches[-1].upper()
        if final_match in valid_labels:
            return final_match

    # --- TẦNG 2: Đầu dòng (I. hoặc I)) ---
    # Rủi ro: Có thể bắt nhầm "I. Giới thiệu"
    # Giải pháp: Regex yêu cầu [A-Z] phải là ký tự in hoa, theo sau là chấm/ngoặc
    # Nếu prompt của bạn tốt (không yêu cầu model in ra dàn ý), tầng này vẫn ổn.
    bullet_matches = re.findall(
        r"(?:^|\n)[\s*#]*([A-Z])[.)](?=\s|$)", 
        clean_response
    )
    
    if bullet_matches:
        final_bullet = bullet_matches[-1].upper()
        if final_bullet in valid_labels:
            return final_bullet

    # --- TẦNG 3: Ký tự đứng một mình ---
    standalone_match = re.search(r"^[\s*#]*([A-Z])[\s*#.]*$", clean_response)
    if standalone_match:
        char = standalone_match.group(1).upper()
        if char in valid_labels:
            return char

    # Fallback
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
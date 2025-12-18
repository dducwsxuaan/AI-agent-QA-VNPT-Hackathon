from typing import TypedDict, List, Optional

class GraphState(TypedDict):
    """
    Trạng thái của đồ thị (State), lưu trữ dữ liệu qua các bước xử lý.
    """
    qid: str
    question: str
    choices: List[str]
    choices_formatted: str  # Chuỗi "A. ... \n B. ..."
    
    # Kết quả từ Router
    genre: str              # math, rag, reading, toxic
    rewrite_query: str
    keywords: str
    
    # Kết quả xử lý
    context: Optional[str]  # Dữ liệu tìm kiếm từ RAG
    answer: str             # Đáp án cuối cùng (A, B, C...) 

def get_choices_from_state(state: GraphState) -> List[str]:
    """Lấy danh sách lựa chọn từ state một cách an toàn."""
    return state.get("choices", [])
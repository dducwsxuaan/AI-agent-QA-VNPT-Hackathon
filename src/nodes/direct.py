from langchain_core.prompts import ChatPromptTemplate
from src.state import GraphState, get_choices_from_state
from src.data_processing.formatting import extract_answer_key
from src.data_processing.answer import extract_answer
from src.data_processing.formatting import format_choices

def reading_node(state: GraphState, llm_large):
    print(f"--- Reading Node: {state['qid']} ---")

    question = state["question"]
    choices_str = state.get("choices_formatted", "")
    
    if not choices_str:
        all_choices = get_choices_from_state(state)
        choices_str = format_choices(all_choices)

    system_prompt = """Bạn là một chuyên gia phân tích văn bản và đọc hiểu. Nhiệm vụ của bạn là đọc kỹ đoạn văn bản được cung cấp và trả lời câu hỏi trắc nghiệm dựa DUY NHẤT vào văn bản trong câu hỏi.

            QUY TẮC CỐT LÕI (TUYỆT ĐỐI TUÂN THỦ):
            1. KHÔNG sử dụng kiến thức bên ngoài xã hội/lịch sử để trả lời. Mọi đáp án phải được chứng minh bằng thông tin nằm trong phần "Văn bản nguồn".
            2. Nếu văn bản nguồn mâu thuẫn với thực tế, hãy ưu tiên văn bản nguồn.
            3. Có thể đáp án không có các chữ cái đứng đầu, thứ tự đáp án sẽ tương đương với thứ tự bảng chữ cái (dòng 1 là A, dòng 2 là B...).

            ĐỊNH DẠNG ĐẦU RA (Suy luận trước, kết luận sau):
            - Bước 1 (Tìm kiếm): Trích dẫn nguyên văn câu/đoạn trong văn bản chứa thông tin cần thiết (Ghi: "Dẫn chứng: ...").
            - Bước 2 (So khớp): So sánh dẫn chứng đó với các lựa chọn để loại trừ đáp án sai.
            - Bước 3 (Kết luận): Dòng cuối cùng BẮT BUỘC ghi đáp án theo mẫu: "Answer: [Đáp án]" (Ví dụ: Answer: A).
            """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Câu hỏi (Question):\n{question}\n\nLựa chọn (Choices):\n{choices}")
    ])
    
    chain = prompt | llm_large
    try:
        response = chain.invoke({
            "question": question,
            "choices": choices_str 
        })

        print(f"🟢 [DEBUG Reading] Response:\n{response.content}\n{'-'*30}")
        
        final_ans = extract_answer(response.content)
        
        return {
            "answer": final_ans,
            "explanation": response.content, 
            "choices_formatted": choices_str 
        }
    except Exception as e:
        print(f"❌ [Reading Node] Error: {e}")
        return {"answer": "A", "explanation": str(e)}
    

def toxic_node(state: GraphState, llm_large):
    print(f"--- Toxic Node: {state['qid']} ---")
    question = state["question"]
    choices_str = state.get("choices_formatted", "")
    
    if not choices_str:
        all_choices = get_choices_from_state(state)
        choices_str = format_choices(all_choices)

    system_prompt = """Bạn là một trợ lý AI hữu ích, câu hỏi trên đã được phân loại là một câu hỏi không nên trả lời(toxic). Bạn hãy chọn câu trả lời
              mà ý nghĩa của nó chỉ ra câu hỏi là một câu nhạy cảm và không trả lời câu hỏi.
              QUY TẮC AN TOÀN VÀ TRUNG THỰC:
              1. Câu hỏi đã được phân loại nhạy cảm, hãy ưu tiên lựa chọn các câu trả lời có ý nghĩa: "Tôi không thể trả lời", "Không có thông tin", hoặc "Câu hỏi nằm ngoài phạm vi".
              2. Có thể đáp án không có các chữ cái đứng đầu, thứ tự đáp án sẽ tương đương với thứ tự bảng chữ cái(ví dụ đáp án đầu là A).
              ĐỊNH DẠNG ĐẦU RA:
              - Dòng cuối cùng BẮT BUỘC phải ghi đáp án ở cuối câu trả lời theo mẫu: "Answer: [Đáp án]" (Ví dụ: Answer: A).
            """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Câu hỏi (Question):\n{question}\n\nLựa chọn (Choices):\n{choices}")
    ])
    
    chain = prompt | llm_large
    try:
        response = chain.invoke({
            "question": question,
            "choices": choices_str
        })
        
        # --- LOGGING RA CONSOLE ---
        print(f"⚠️ [DEBUG Toxic] Response:\n{response.content}\n{'-'*30}")
        
        final_ans = extract_answer(response.content)
        
        return {
            "answer": final_ans,
            "explanation": response.content, # Lưu lại nội dung gốc
            "choices_formatted": choices_str
        }

    except Exception as e:
        print(f"❌ [Toxic Node] Error: {e}")
        return {"answer": "A", "explanation": str(e)}

"""
Math/Logic Solver Node: ReAct Pattern với Code Execution an toàn.
Tích hợp validation, tự động sửa lỗi và vòng lặp phản hồi (Feedback Loop).
"""
import re
import math
from typing import List, Optional, Tuple, Any

# LangChain Imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_experimental.utilities import PythonREPL

# Internal Imports
from src.state import GraphState, get_choices_from_state
from src.data_processing.formatting import format_choices
from src.data_processing.answer import extract_answer
from src.utils.logging import print_log

_python_repl = PythonREPL()

# --- HELPER FUNCTIONS (NÂNG CẤP) ---

def _clean_code(code: str) -> str:
    """Làm sạch code markdown."""
    code = re.sub(r"^```(?:python)?\s*", "", code, flags=re.MULTILINE | re.IGNORECASE)
    code = re.sub(r"```\s*$", "", code, flags=re.MULTILINE)
    return code.strip()

def _extract_code(text: str) -> Optional[str]:
    """Trích xuất block code ưu tiên python."""
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1)
    
    match_generic = re.search(r"```\n(.*?)\n```", text, re.DOTALL)
    if match_generic: return match_generic.group(1)
    
    return None

def _validate_code_syntax(code: str) -> Tuple[bool, str]:
    """[NEW] Kiểm tra cú pháp code trước khi chạy để tránh crash."""
    try:
        compile(code, "<string>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, str(e)

def _is_placeholder_code(code: str) -> bool:
    """[NEW] Phát hiện code rác, chưa hoàn thiện (chứa '...' hoặc quá ngắn)."""
    if not code or len(code.strip()) < 10:
        return True
    if "..." in code or "pass" == code.strip():
        return True
    # Kiểm tra placeholder kiểu {var} nhưng không phải f-string
    if re.search(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", code) and not re.search(r'f["\']', code):
        return True
    return False

def _auto_add_print(code: str) -> str:
    """[NEW] Nếu code không có lệnh print, tự động print biến cuối cùng."""
    if "print" not in code:
        lines = code.splitlines()
        # Tìm dòng code cuối cùng có nội dung (không phải comment)
        valid_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
        if valid_lines:
            last_line = valid_lines[-1]
            # Nếu là phép gán (x = 10), print biến đó
            if "=" in last_line:
                var_name = last_line.split("=")[0].strip()
                code += f"\nprint({var_name})"
    return code

# --- MAIN NODE ---

def math_logic_node(state: GraphState, llm_large) -> dict:
    qid = state.get("qid", "unknown")
    print_log(f"--- Math Node: {qid} ---")

    question_text = state["question"]
    # Xử lý choices an toàn
    choices_str = state.get("choices_formatted", "")
    if not choices_str:
        all_choices = get_choices_from_state(state)
        choices_str = format_choices(all_choices)

    # Prompt chi tiết giữ nguyên chiến lược Symbolic/Numeric
    system_content = """Bạn là một chuyên gia lập trình Python xuất sắc, chuyên giải quyết các bài toán trắc nghiệm Toán học và Logic bằng phương pháp tính toán máy tính (Computational Thinking).

    NHIỆM VỤ:
    Viết code Python chính xác để tính toán và in ra đáp án đúng nhất (A, B, C, hoặc D).

    ---
    PHẦN 1: QUY TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ)
    1. Tên biến an toàn: TUYỆT ĐỐI KHÔNG dùng từ khóa Python (`min`, `max`, `sum`, `lambda`, `return`, `class`...) làm tên biến. Hãy dùng `val_x`, `total_sum`, `min_val`...
    2. Xử lý đơn vị: Nếu đáp án có đơn vị (ví dụ: "50 km/h", "10%", "$500"), hãy loại bỏ đơn vị, chỉ lấy giá trị số thực để tính toán.
    3. KHÔNG CHỌN BỪA: Không được in đáp án dựa trên suy đoán. Phải dùng thuật toán so sánh sai số nhỏ nhất (`min_diff`) để tìm đáp án khớp nhất.
    4. Định nghĩa Options: BẮT BUỘC phải tạo dictionary `options` chứa giá trị số của 4 lựa chọn.

    ---
    PHẦN 2: CHIẾN LƯỢC GIẢI QUYẾT

    [TRƯỜNG HỢP 1]: BÀI TOÁN ĐẠI SỐ / CÔNG THỨC (SYMBOLIC)
    - Dấu hiệu: Đáp án chứa biến số (x, y, m, g, t...).
    - Phương pháp: "Thử số ngẫu nhiên" (Numerical Substitution).
      1. Gán giá trị số thực lẻ (tránh 0, 1, số nguyên chẵn) cho các biến (ví dụ: x=2.1, y=0.7).
      2. Tính giá trị số của biểu thức đề bài (`target_val`).
      3. Tính giá trị số của từng đáp án trong `options`.
      4. Chọn đáp án có `abs(target_val - option_val)` nhỏ nhất.

    [TRƯỜNG HỢP 2]: BÀI TOÁN SỐ HỌC (NUMERIC)
    - Dấu hiệu: Đáp án là các con số cụ thể.
    - Phương pháp: Tính toán trực tiếp.
      1. Viết logic tính ra kết quả `result`.
      2. So sánh `result` với các giá trị trong `options` để tìm ra số gần nhất (xử lý sai số làm tròn).

    ---
    PHẦN 3: CODE MẪU THAM KHẢO (HÃY VIẾT THEO CẤU TRÚC NÀY)

    ### Mẫu A: Giải bài toán Công thức (Dùng Sympy & Numerical Substitution)
    ```python
    import sympy as sp

    # 1. Định nghĩa biến và gán giá trị thử (Tránh 0, 1)
    vals = {'m': 2.3, 'g': 9.81, 'h': 10.5, 'v': 5.2}
    m, g, h, v = sp.symbols('m g h v')

    # 2. Biểu thức mục tiêu (Đề bài)
    target_expr = m*g*h + 0.5*m*v**2
    target_val = float(target_expr.subs(vals))

    # 3. Các lựa chọn (Lưu ý: định nghĩa dictionary options là BẮT BUỘC)
    options = {
        "A": m*g*h,
        "B": 0.5*m*v**2,
        "C": m*g*h + 0.5*m*v**2,
        "D": m*g*h - 0.5*m*v**2
    }

    # 4. So sánh tìm sai số nhỏ nhất (Robust Matching)
    best_key = None
    min_diff = float('inf')

    for key, expr in options.items():
        try:
            opt_val = float(expr.subs(vals))
            diff = abs(target_val - opt_val)
            if diff < min_diff:
                min_diff = diff
                best_key = key
        except: continue

    print(f"Answer: {best_key}")
    ```

    ### Mẫu B: Giải bài toán Số học (Tính toán trực tiếp)
    ```python
    import math

    # 1. Định nghĩa options (Lọc bỏ đơn vị % hoặc text)
    options = {"A": 10.5, "B": 20.0, "C": 15.2, "D": 30.0}

    # 2. Tính toán logic
    # Ví dụ: Tìm nồng độ pH
    H_conc = 1e-5
    result = -math.log10(H_conc)

    # 3. So sánh (Tìm key có giá trị gần result nhất)
    found = False
    # Cách 1: So sánh chính xác (với dung sai nhỏ)
    for key, val in options.items():
        if math.isclose(result, val, rel_tol=1e-3):
            print(f"Answer: {key}")
            found = True
            break

    # Cách 2: Tìm min diff nếu không khớp chính xác
    if not found:
        best = min(options, key=lambda k: abs(options[k]-result))
        print(f"Answer: {best}")
    ```

    LƯU Ý CUỐI CÙNG:
    - Luôn phải có bước định nghĩa `options`.
    - Luôn kết thúc bằng `print(f"Answer: {key}")`.
    """

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=f"Câu hỏi: {question_text}\nLựa chọn:\n{choices_str}\n\nHãy phân tích và viết code giải bài này.")
    ]

    max_retries = 5 # Tăng số lượt để đủ vòng lặp ReAct
    final_ans = "A" 

    for attempt in range(max_retries):
        print_log(f"        [Math] Step {attempt + 1}/{max_retries}...")

        # 1. GỌI LLM
        try:
            ai_msg = llm_large.invoke(messages)
            content = ai_msg.content
            messages.append(ai_msg)
            
            # [LOGGING] In ra suy luận của model
            print_log(f"🟢 [LLM Response]:\n{content[:500]}..." if len(content) > 500 else f"🟢 [LLM Response]:\n{content}")
            
        except Exception as e:
            print_log(f"        [Math] LLM Error: {e}")
            break

        # 2. KIỂM TRA ĐÁP ÁN SỚM
        # Nếu model đã tự tin chốt đáp án (có từ khóa Answer:) và không cần code nữa
        step_ans = extract_answer(content)
        # Regex kiểm tra xem có dòng "Answer: X" rõ ràng không
        has_explicit_answer = re.search(r"(?:Answer|Đáp án)[:\s]+([A-Z])", content, re.IGNORECASE)
        
        if has_explicit_answer and step_ans:
            print_log(f"        [Math] Found explicit answer: {step_ans}")
            return {"answer": step_ans, "raw_response": content}

        # 3. TRÍCH XUẤT CODE
        code_block = _extract_code(content)

        # Nếu không có code
        if not code_block:
            if attempt < max_retries - 1:
                print_log("        [Math] No code found. Reminding...")
                messages.append(HumanMessage(content="Bạn chưa viết code Python tính toán. Hãy viết code trong block ```python ... ``` ngay."))
                continue
            else:
                # Lượt cuối chấp nhận text reasoning
                final_ans = step_ans
                break

        code_block = _clean_code(code_block)

        # 4. VALIDATION & SANITIZATION (BƯỚC MỚI QUAN TRỌNG)
        
        # Check code rác
        if _is_placeholder_code(code_block):
            print_log("        [Math] Placeholder code detected.")
            messages.append(HumanMessage(content="Code của bạn chưa hoàn thiện hoặc chứa placeholder (...). Hãy viết code đầy đủ, chạy được."))
            continue
            
        # Check cú pháp (Syntax Check)
        is_valid, syntax_err = _validate_code_syntax(code_block)
        if not is_valid:
            print_log(f"        [Math] Syntax Error: {syntax_err}")
            messages.append(HumanMessage(content=f"Code lỗi cú pháp (SyntaxError): {syntax_err}. Hãy sửa lại code ngay."))
            continue

        # Auto-add Print
        code_block = _auto_add_print(code_block)

        # 5. THỰC THI CODE
        print_log(f"        [Math] Executing Code...")
        try:
            exec_result = _python_repl.run(code_block)
            exec_result = exec_result.strip() if exec_result else "No output."
            print_log(f"🟡 [Code Output]: {exec_result}")

            # 6. FEEDBACK LOOP (VÒNG LẶP PHẢN HỒI)
            # Thay vì tự parse output, gửi lại cho LLM để nó tự kết luận
            # Đây là mấu chốt của ReAct: Model quan sát kết quả của chính mình
            
            feedback_msg = (
                f"Kết quả chạy code là:\n{exec_result}\n\n"
                "Dựa vào kết quả trên, hãy so sánh với các Lựa chọn (A, B, C, D) và đưa ra kết luận cuối cùng.\n"
                "BẮT BUỘC kết thúc câu trả lời bằng dòng: 'Answer: X' (X là ký tự đáp án)."
            )
            messages.append(HumanMessage(content=feedback_msg))
            
            # Tiếp tục vòng lặp để LLM đọc feedback này và trả lời ở lượt sau
            continue

        except Exception as e:
            # Xử lý Runtime Error
            error_msg = str(e)
            print_log(f"        [Math] Runtime Error: {error_msg}")
            
            # Gợi ý sửa lỗi thông minh
            fix_prompt = f"Code bị lỗi Runtime: {error_msg}. Hãy sửa lại code."
            
            if "name 'options' is not defined" in error_msg:
                fix_prompt += "\nLÝ DO: Bạn quên khai báo biến `options`. Hãy thêm `options = {'A':..., 'B':...}`."
            
            messages.append(HumanMessage(content=fix_prompt))
            continue

    # --- 7. FALLBACK FINAL ---
    print_log("        [Math] Retries exhausted. Using Fallback logic.")

    if len(messages) > 2:
        try:
            # Lần chốt hạ cuối cùng: Ép chọn đáp án
            print_log("        [Math] Asking for final conclusion...")
            final_response = llm_large.invoke(messages + [HumanMessage(content="Dựa trên tất cả các bước trên, hãy chốt đáp án cuối cùng là A, B, C hay D? Chỉ trả về: Answer: X")])
            final_ans = extract_answer(final_response.content)
            print_log(f"        [Math] Final Fallback Ans: {final_ans}")
            
            full_log = "\n".join([m.content for m in messages if isinstance(m, AIMessage)])
            return {"answer": final_ans, "raw_response": full_log}
            
        except Exception as e:
            print_log(f"        [Math] Fallback failed: {e}")

    return {"answer": final_ans, "raw_response": "Execution failed."}
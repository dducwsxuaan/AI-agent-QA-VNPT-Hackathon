import json
import re
from langchain_core.prompts import ChatPromptTemplate
from src.state import GraphState

def router_node(state: GraphState, llm_small):
    """Phân loại câu hỏi và rewrite."""
    query = state["question"]
    
    # Load prompt template (thực tế nên load từ file .j2)
    prompt_template = """Bạn là một trợ lý AI chuyên xử lý ngôn ngữ cho hệ thống trả lời câu hỏi.
              Nhiệm vụ:
              1. Phân loại câu hỏi dựa trên các loại sau đây(genre):
                  - math: Câu hỏi cần tính toán toán học, logic hoặc lập trình.
                  - reading: Câu hỏi đọc hiểu dựa trên đoạn văn được cung cấp ngay trong đề.
                  - rag: Câu hỏi cần tra cứu kiến thức thực tế (Lịch sử, Địa lý, Văn hóa, Xã hội, Chính Trị,...).
                  - toxic: Câu hỏi yêu cầu hướng dẫn làm việc phi pháp(trốn thuế, chế tạo vũ khí, tấn công mạng...), những câu hỏi mang tính chống phá nhà nước việt nam(vi phạm chủ quyền, đường lối chính sách của đảng) .
              Nếu câu hỏi thuộc loại rag hãy làm tiếp các nhiệm vụ sau:
              1.1. Viết lại câu hỏi (rewrite): Rõ nghĩa, đầy đủ chủ ngữ vị ngữ, sửa lỗi chính tả nếu có.
              1.2. Trích xuất từ khóa (keywords): Tên riêng, địa danh, thuật ngữ quan trọng.

              QUY ĐỊNH BẮT BUỘC:
              - Chỉ trả về duy nhất một JSON object hợp lệ.
              - Không bao gồm markdown tick (```json ... ```).
              - Không giải thích gì thêm.

              Ví dụ 1:
              User: Bác Hồ sinh năm nào?
              Output: {{
                  "genre": "rag",
                  "rewrite": "Chủ tịch Hồ Chí Minh sinh vào năm nào?",
                  "keywords": "Hồ Chí Minh, năm sinh"
              }}

              Ví dụ 2:
              User: Một sợi dây có chiều dài $ L $ và mật độ khối lượng tuyến tính $ \\mu $ được giữ cố định ở cả hai đầu và chịu lực căng $ T $. Sợi dây được kích thích dao động sao cho tạo ra một sóng dừng với $ n $ nút (bao gồm cả các đầu mút). Nếu lực căng của sợi dây được tăng lên gấp 4 lần, tần số cơ bản $ f_1 $ của sợi dây thay đổi như thế nào?
              Output: {{
                  "genre": "math"
              }}

              Ví dụ 3:
              User: Chiến thắng ĐBP trên không diễn ra ở đâu?
              Output: {{
                  "genre": "rag",
                  "rewrite": "Chiến dịch Điện Biên Phủ trên không diễn ra tại địa điểm nào?",
                  "keywords": "Điện Biên Phủ trên không, địa điểm"
              }}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("user", "{question}")
    ])
    
    chain = prompt | llm_small
    response = chain.invoke({"question": query}).content
    
    try:
        # Logic parse JSON từ Notebook
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx : end_idx + 1].replace("\\'", "'")
            json_str = re.sub(r',\s*}', '}', json_str)
            data = json.loads(json_str)
            
            return {
                "rewrite_query": data.get("rewrite", query),
                "keywords": data.get("keywords", query),
                "genre": data.get("genre", "rag")
            }
    except:
        pass
        
    return {"rewrite_query": query, "keywords": query, "genre": "rag"} 

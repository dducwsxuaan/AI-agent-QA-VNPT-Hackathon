from langgraph.graph import StateGraph, END
from src.state import GraphState
from src.nodes.router import router_node
from src.nodes.rag import rag_node
from src.nodes.math_solver import math_logic_node
from src.nodes.direct import reading_node, toxic_node
from src.config import settings
from src.utils.llm_wrappers import VNPTChatModel

# 1. Khởi tạo Models với config đầy đủ từ settings
# LƯU Ý: Phải truyền đúng tên tham số (token_id, token_key...)
llm_small = VNPTChatModel(
    model_name="vnptai-hackathon-small",
    token_id=settings.VNPT_TOKEN_ID_SMALL,
    token_key=settings.VNPT_API_KEY_SMALL,
    access_token=settings.VNPT_ACCESS_TOKEN_SMALL,
    temperature=0.1
)

llm_large = VNPTChatModel(
    model_name="vnptai-hackathon-large",
    token_id=settings.VNPT_TOKEN_ID_LARGE,
    token_key=settings.VNPT_API_KEY_LARGE,
    access_token=settings.VNPT_ACCESS_TOKEN_LARGE,
    temperature=0.1
)

# 2. Định nghĩa điều hướng
def route_condition(state):
    return state["genre"]

# 3. Xây dựng Graph
def build_graph():
    workflow = StateGraph(GraphState)
    
    # Add Nodes
    workflow.add_node("router", lambda x: router_node(x, llm_small))
    
    # Lưu ý: Node RAG cần truyền thêm db ở main.py (inject dependencies)
    # Ở đây ta định nghĩa node wrapper nhận db từ lambda
    # Tuy nhiên để đơn giản, ta sẽ để logic inject db ở main.py khi compile graph
    # Node define ở đây là placeholder
    workflow.add_node("rag", lambda x: rag_node(x, llm_large)) 
    
    workflow.add_node("math", lambda x: math_logic_node(x, llm_large))
    workflow.add_node("reading", lambda x: reading_node(x, llm_large))
    workflow.add_node("toxic", lambda x: toxic_node(x, llm_large))
    
    # Edges
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        route_condition,
        {
            "rag": "rag",
            "math": "math",
            "reading": "reading",
            "toxic": "toxic"
        }
    )
    
    workflow.add_edge("rag", END)
    workflow.add_edge("math", END)
    workflow.add_edge("reading", END)
    workflow.add_edge("toxic", END)
    
    # Chỉ trả về workflow chưa compile để main.py có thể chỉnh sửa node nếu cần
    # Hoặc compile luôn nếu main.py đã xử lý inject
    return workflow.compile()
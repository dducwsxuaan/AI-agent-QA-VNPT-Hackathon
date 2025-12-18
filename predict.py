import sys
import json
import csv
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import settings
from src.utils.llm_wrappers import VNPTChatModel, VNPTEmbeddingWrapper
from src.database.hybrid_qdrant import QdrantHybridDB, HybridEncoder
from src.data_processing.formatting import format_choices
from src.graph import build_graph, llm_small, llm_large # Import graph đã compile

INPUT_PATH = "/code/private_test.json"
OUTPUT_PATH = "/code/submission.csv"



def main():
    parser = argparse.ArgumentParser(description="VNPT RAG Pipeline")
    parser.add_argument("--input", type=str, default=INPUT_PATH, help="Path to input JSON")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="Path to output CSV")
    args = parser.parse_args()

    # 1. Khởi tạo DB (Lazy loading)
    print("🔌 Kết nối Database...")
    embed_model = VNPTEmbeddingWrapper(
        token_id=settings.VNPT_TOKEN_ID_EMBED,
        token_key=settings.VNPT_API_KEY_EMBED,
        access_token=settings.VNPT_ACCESS_TOKEN_EMBED
    )
    encoder = HybridEncoder(embed_model)
    db = QdrantHybridDB(path=str(settings.DB_PATH))
    db.set_encoder(encoder) # Quan trọng: Set encoder để search

    # 2. Cấu hình LLM cho Graph nodes (Inject DB vào node RAG)
    # Lưu ý: Trong file src/graph.py, chúng ta cần sửa lại một chút để truyền db vào node RAG
    # Hoặc đơn giản là dùng biến global/closure. Ở đây tôi đề xuất inject khi chạy graph.
    
    # Re-compile graph với DB instance thực tế
    from src.nodes.rag import rag_node
    from src.nodes.router import router_node
    from src.nodes.math_solver import math_logic_node
    from src.nodes.direct import reading_node, toxic_node
    from src.state import GraphState
    from langgraph.graph import StateGraph, END
    
    workflow = StateGraph(GraphState)
    workflow.add_node("router", lambda x: router_node(x, llm_small))
    
    # QUAN TRỌNG: Truyền llm_small vào để làm Reranking
    workflow.add_node("rag", lambda x: rag_node(x, llm_large, llm_small, db)) 
    
    workflow.add_node("math", lambda x: math_logic_node(x, llm_large))
    workflow.add_node("reading", lambda x: reading_node(x, llm_large))
    workflow.add_node("toxic", lambda x: toxic_node(x, llm_large))
    
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        lambda x: x["genre"],
        {"rag": "rag", "math": "math", "reading": "reading", "toxic": "toxic"}
    )
    workflow.add_edge("rag", END)
    workflow.add_edge("math", END)
    workflow.add_edge("reading", END)
    workflow.add_edge("toxic", END)
    
    app = workflow.compile()

    # 3. Xử lý Input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ File {input_path} không tồn tại.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    print(f"🚀 Bắt đầu xử lý {len(data)} câu hỏi...")

    for item in data:
        qid = item.get("id") or item.get("qid")
        question = item.get("question")
        choices = item.get("choices")
        
        # Tạo input state
        initial_state = {
            "qid": qid,
            "question": question,
            "choices": choices,
            "choices_formatted": format_choices(choices),
            # Các trường khác để None hoặc default
        }
        
        try:
            output_state = app.invoke(initial_state)
            ans = output_state.get("answer", "A")
            print(f"[{qid}] Genre: {output_state.get('genre')} | Ans: {ans}")
            results.append([qid, ans])
        except Exception as e:
            print(f"❌ Lỗi câu {qid}: {e}")
            results.append([qid, "A"]) # Fallback

    # 4. Lưu kết quả
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'answer']) # Header submission
        writer.writerows(results)
    
    print(f"✅ Đã lưu kết quả tại {output_path}")

if __name__ == "__main__":
    main()
"""RAG node for knowledge-based question answering with Retrieve & Rerank."""

import re
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.state import GraphState, get_choices_from_state
from src.database.hybrid_qdrant import QdrantHybridDB
from src.data_processing.answer import extract_answer # Import hàm extract mới
from src.data_processing.formatting import format_choices


def _rerank_documents(query: str, docs: list, llm_small, top_k: int = 5) -> list:
    """Hàm phụ trợ: Sắp xếp lại văn bản bằng mô hình nhỏ."""
    if len(docs) <= top_k:
        return docs
    
    doc_list = ""
    for i, doc in enumerate(docs):
        content_preview = doc.page_content[:300].replace("\n", " ")
        doc_list += f"[{i}] {content_preview}...\n\n"
    
    rerank_system = (
        "Bạn là chuyên gia đánh giá độ liên quan của văn bản. "
        "Nhiệm vụ: Chọn ra các đoạn văn bản LIÊN QUAN NHẤT với câu hỏi.\n"
        "Chỉ trả về danh sách các số ID (ví dụ: 0, 3, 5), không giải thích gì thêm."
    )
    
    rerank_user = (
        f"Câu hỏi: {query}\n\n"
        f"Các đoạn văn bản:\n{doc_list}\n"
        f"Hãy chọn {top_k} đoạn văn bản LIÊN QUAN NHẤT. "
        f"Trả về danh sách ID (số từ 0 đến {len(docs)-1}), cách nhau bởi dấu phẩy."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", rerank_system),
        ("human", rerank_user),
    ])
    
    try:
        chain = prompt | llm_small
        response = chain.invoke({})
        content = response.content.strip()
        
        # Parse ID
        selected_ids = []
        numbers = re.findall(r'\d+', content)
        for num_str in numbers:
            idx = int(num_str)
            if 0 <= idx < len(docs) and idx not in selected_ids:
                selected_ids.append(idx)
                if len(selected_ids) >= top_k:
                    break
        
        if selected_ids:
            reranked = [docs[i] for i in selected_ids]
            # print(f"        [Rerank] Selected {len(reranked)} docs.")
            return reranked
        
        return docs[:top_k]
        
    except Exception as e:
        print(f"        [Rerank] Error: {e}")
        return docs[:top_k]

def rag_node(state: GraphState, llm_large, llm_small, vector_db: QdrantHybridDB):
    print(f"--- RAG Node: {state['qid']} ---")
    
    query_dense = state.get("rewrite_query", state["question"])
    query_sparse = state.get("keywords", state["question"])
    # raw_choices = state.get("choices", [])

    choices_str = state.get("choices_formatted", "")
    if not choices_str:
        all_choices = get_choices_from_state(state)
        choices_str = format_choices(all_choices)

    # choices_formatted = format_choices(raw_choices) if raw_choices else ""
    
    # 1. Retrieve
    docs = vector_db.hybrid_search(
        query_dense=query_dense,
        query_sparse=query_sparse,
        top_k=20 
    )
    
    # 2. Rerank
    if docs:
        reranked_docs = _rerank_documents(
            query=state["question"], 
            docs=docs, 
            llm_small=llm_small, 
            top_k=8 
        )
    else:
        reranked_docs = []

    context_text = "\n\n".join([doc.page_content for doc in reranked_docs])
    
    # 3. Generate Answer
    system_prompt = """Bạn là một trợ lý AI hữu ích, nhiệm vụ của bạn là trả lời câu hỏi trắc nghiệm dựa trên thông tin được cung cấp.
    QUY TẮC AN TOÀN VÀ TRUNG THỰC:
    1. Chỉ sử dụng thông tin trong phần "Ngữ cảnh" (Context) bên dưới để trả lời.
    2. Nếu không tìm thấy thông tin trong ngữ cảnh, hãy dùng kiến thức của bạn nhưng phải ưu tiên ngữ cảnh.
    3. Dòng cuối cùng BẮT BUỘC phải ghi đáp án ở cuối câu trả lời theo mẫu: "Answer: [Đáp án]" (Ví dụ: Answer: A).
    4. Nếu như bạn không thể tìm ra đáp án trong các đáp án đã cho, hãy suy luận lại kỹ 1 lần nữa. Cố gắng xem xét thật kỹ các đáp án. Trong trường hợp đã thử rất nhiều mà không thể tìm ra, hãy chọn đáp án có khả năng cao nhất.
    5. Tuyệt đối không được bịa đặt đáp án nếu không chắc chắn.
    
    """
    
    user_prompt = """Ngữ cảnh (Context):
    {context}

    Câu hỏi (Question):
    {question}
    Lựa chọn (Choices):
    {choices_str}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    
    chain = prompt | llm_large
    
    try:
        response = chain.invoke({
            "context": context_text,
            "question": state["question"],
            "choices_str": choices_str
        })
        
        # --- DEBUG QUAN TRỌNG: In ra xem model nói gì ---
        print(f"🔴 [DEBUG LLM] Raw content: {response.content}") 
        # ------------------------------------------------
        
        final_answer = extract_answer(response.content)
        return {"answer": final_answer, "context": context_text}
        
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return {"answer": "A", "context": context_text}
import sys
import glob
from pathlib import Path
from tqdm import tqdm


### IF INGESTING MORE DATA, change the directory at the line: 
# corpus_dir = settings.DATA_DIR / "crawl"


# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.utils.llm_wrappers import VNPTEmbeddingWrapper
from src.database.hybrid_qdrant import QdrantHybridDB, HybridEncoder
from src.database.splitter import SemanticSplitter

def main():
    print("🚀 Bắt đầu nạp 239 file (Chế độ Batch Size = 32)...")
    
    # 1. Setup
    embed_model = VNPTEmbeddingWrapper(
        token_id=settings.VNPT_TOKEN_ID_EMBED,
        token_key=settings.VNPT_API_KEY_EMBED,
        access_token=settings.VNPT_ACCESS_TOKEN_EMBED
    )
    splitter = SemanticSplitter(embed_model)
    encoder = HybridEncoder(embed_model)
    
    # Kết nối DB (Dữ liệu cũ vẫn được giữ nguyên)
    db = QdrantHybridDB(path=str(settings.DB_PATH))
    
    # 2. Tìm file
    corpus_dir = settings.DATA_DIR / "crawl" 
    if not corpus_dir.exists():
        corpus_dir = settings.DATA_DIR

    files = list(corpus_dir.rglob("*.txt"))
    
    if not files:
        print(f"❌ Không tìm thấy file .txt nào tại {corpus_dir}")
        return

    print(f"📂 Tìm thấy {len(files)} file. Đang xử lý lần lượt...")

    # 3. Cấu hình Batch
    BATCH_SIZE = 32  # <-- Cấu hình theo yêu cầu của bạn
    
    current_batch = []
    total_chunks_indexed = 0
    
    # Thanh hiển thị tiến trình (Progress Bar)
    pbar = tqdm(files, desc="Đang xử lý")
    
    for file_path in pbar:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if text.strip():
                # Chia nhỏ văn bản thành các chunk
                file_chunks = splitter.split_text(text)
                
                # Thêm vào hàng đợi (queue)
                current_batch.extend(file_chunks)
                
                # --- LOGIC BATCH 32 ---
                # Khi hàng đợi gom đủ hoặc hơn 32 chunks thì đẩy lên ngay
                while len(current_batch) >= BATCH_SIZE:
                    # Lấy đúng 32 chunks đầu tiên để xử lý
                    batch_to_upload = current_batch[:BATCH_SIZE]
                    # Phần dư giữ lại cho đợt sau
                    current_batch = current_batch[BATCH_SIZE:]
                    
                    # Upload
                    db.insert_batch(batch_to_upload, encoder)
                    total_chunks_indexed += len(batch_to_upload)
                    
                    # Update thông báo
                    pbar.set_postfix({
                        "Batch": "32", 
                        "Total Saved": total_chunks_indexed
                    })
                    
        except Exception as e:
            # Nếu 1 file lỗi thì bỏ qua, không dừng chương trình
            # print(f"Lỗi file {file_path.name}: {e}")
            pass

    # 4. Xử lý phần dư cuối cùng (nếu còn < 32 chunks)
    if current_batch:
        db.insert_batch(current_batch, encoder)
        total_chunks_indexed += len(current_batch)

    db.close()
    print(f"\n✅ HOÀN TẤT! Tổng cộng đã index {total_chunks_indexed} chunks từ {len(files)} files.")

if __name__ == "__main__":
    main()
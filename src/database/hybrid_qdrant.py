import os
import zlib
import re
import uuid
from typing import List
from qdrant_client import QdrantClient, models
from langchain_core.documents import Document

class HybridEncoder:
    def __init__(self, dense_model):
        self.dense_model = dense_model

    def _hash_text(self, text: str) -> int:
        return zlib.crc32(text.encode('utf-8')) % 20000

    def encode_dense(self, texts: List[str]):
        return self.dense_model.encode(texts)

    def encode_sparse_query(self, keyword_string: str):
        indices = [self._hash_text(k.strip().lower()) for k in re.split(r'[,;]', keyword_string) if k.strip()]
        return indices, [1.0] * len(indices)

    def encode_sparse_docs(self, documents: List[str]):
        results = []
        for doc in documents:
            words = re.split(r'[, \.;]+', doc.lower())
            indices = [self._hash_text(w) for w in words if w.strip()]
            results.append((list(set(indices)), [1.0] * len(set(indices))))
        return results

class QdrantHybridDB:
    def __init__(self, path: str = None):
        if path:
            os.makedirs(path, exist_ok=True)
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")
        
        self.collection_name = "vnpt_rag_final"
        self.internal_encoder = None

    def set_encoder(self, encoder):
        self.internal_encoder = encoder

    # --- ĐÂY LÀ HÀM BẠN ĐANG THIẾU ---
    def ensure_collection(self):
        """Đảm bảo Collection tồn tại"""
        if not self.client.collection_exists(self.collection_name):
            print(f"⚡ Tạo mới Collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams()}
            )

    def insert_batch(self, documents: List[str], encoder: HybridEncoder):
        """Thêm một lô (batch) dữ liệu vào DB (Upsert)"""
        self.internal_encoder = encoder
        self.ensure_collection() 

        if not documents: return

        # Encode
        dense_vecs = encoder.encode_dense(documents)
        sparse_vecs = encoder.encode_sparse_docs(documents)

        points = []
        for i, doc in enumerate(documents):
            sp_idx, sp_val = sparse_vecs[i]
            # Tạo ID ngẫu nhiên để không bị trùng lặp
            point_id = str(uuid.uuid4()) 
            
            points.append(models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vecs[i],
                    "sparse": models.SparseVector(indices=sp_idx, values=sp_val)
                },
                payload={"page_content": doc}
            ))
        
        # Upsert: Thêm mới hoặc cập nhật
        self.client.upsert(self.collection_name, points)
    # ---------------------------------

    def hybrid_search(self, query_dense, query_sparse, top_k=5):
        if not self.internal_encoder:
            raise ValueError("Encoder not set")
            
        q_dense_vec = self.internal_encoder.encode_dense([query_dense])[0]
        q_sp_idx, q_sp_val = self.internal_encoder.encode_sparse_query(query_sparse)

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=q_dense_vec, using="dense", limit=top_k*2),
                models.Prefetch(query=models.SparseVector(indices=q_sp_idx, values=q_sp_val), using="sparse", limit=top_k*2)
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True
        )
        return [Document(page_content=p.payload['page_content']) for p in results.points]

    def close(self):
        self.client.close()
import re
import numpy as np
from typing import List
from underthesea import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSplitter:
    def __init__(self, embed_model, threshold=0.6):
        self.embed_model = embed_model
        self.threshold = threshold

    def _clean_text(self, text):
        text = re.sub(r'<[^>]+>','', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def split_text(self, text: str) -> List[str]:
        # 1. Tách câu cơ bản (dùng thư viện tiếng Việt)
        sentences = sent_tokenize(self._clean_text(text))
        if not sentences: return []

        # 2. Encode để tính toán vector
        embeddings = self.embed_model.encode(sentences)

        # 3. Gom nhóm (Semantic Chunking)
        chunks = [[sentences[0]]]
        for i in range(1, len(sentences)):
            current_emb = np.array(embeddings[i]).reshape(1, -1)
            prev_emb = np.array(embeddings[i-1]).reshape(1, -1)

            # Tính độ tương đồng giữa 2 câu liền kề
            sim = cosine_similarity(prev_emb, current_emb)[0][0]

            if sim >= self.threshold:
                chunks[-1].append(sentences[i]) # Gộp vào chunk cũ
            else:
                chunks.append([sentences[i]])   # Tạo chunk mới

        return [' '.join(chunk) for chunk in chunks] 

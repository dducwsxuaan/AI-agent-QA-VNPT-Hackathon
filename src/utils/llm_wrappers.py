import requests
import time
from typing import List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration

# --- Phần VNPTChatModel giữ nguyên ---
class VNPTChatModel(BaseChatModel):
    base_url: str = "https://api.idg.vnpt.vn/data-service/v1/chat/completions"
    model_name: str
    token_id: str
    token_key: str
    access_token: str
    temperature: float = 0.1
    max_tokens: int = 1024
    max_retries: int = 3

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        vnpt_messages = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage): role = "system"
            elif isinstance(msg, AIMessage): role = "assistant"
            vnpt_messages.append({"role": role, "content": msg.content})

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Token-id": self.token_id,
            "Token-key": self.token_key,
            "Content-Type": "application/json"
        }
        
        normalized_model = self.model_name.replace("-", "_")
        payload = {
            "model": normalized_model,
            "messages": vnpt_messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "top_p": 1.0,
            "n": 1
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(f"{self.base_url}/{self.model_name}", headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
            except Exception as e:
                time.sleep(1 * (attempt + 1))
        
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Error"))])

    @property
    def _llm_type(self) -> str:
        return "vnpt-chat-model"


# --- Phần VNPTEmbeddingWrapper được nâng cấp ---
class VNPTEmbeddingWrapper:
    def __init__(self, token_id, token_key, access_token, max_workers=10):
        self.api_url = "https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding"
        self.headers = {
            'Authorization': f'Bearer {access_token}',
            'Token-id': token_id,
            'Token-key': token_key,
            'Content-Type': 'application/json'
        }
        self.model_name = "vnptai_hackathon_embedding"
        self.embedding_dim = 1024
        self.max_workers = max_workers # Số luồng chạy song song (Recommend: 10-20)

    def _encode_single(self, text: str) -> List[float]:
        """Hàm gọi API cho 1 text duy nhất"""
        if not text or not text.strip():
            return [0.0] * self.embedding_dim
            
        payload = {
            "model": self.model_name, 
            "input": text, 
            "encoding_format": "float"
        }
        
        try:
            # Thêm retry nhẹ nếu lỗi mạng
            for _ in range(3):
                try:
                    response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
                    if response.status_code == 200:
                        return response.json()['data'][0]['embedding']
                    elif response.status_code == 429: # Rate limit
                        time.sleep(2)
                        continue
                    else:
                        break
                except requests.exceptions.RequestException:
                    time.sleep(1)
            
            # Nếu vẫn lỗi thì trả về vector 0
            return [0.0] * self.embedding_dim
        except:
            return [0.0] * self.embedding_dim

    def encode(self, sentences: List[str]) -> List[List[float]]:
        """Xử lý song song danh sách input"""
        if isinstance(sentences, str): sentences = [sentences]
        
        # Tạo danh sách kết quả có thứ tự tương ứng với input
        results = [None] * len(sentences)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Map index với future để đảm bảo thứ tự
            future_to_idx = {
                executor.submit(self._encode_single, text): i 
                for i, text in enumerate(sentences)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    embedding = future.result()
                    results[idx] = embedding
                except Exception as e:
                    results[idx] = [0.0] * self.embedding_dim
        
        return results
FROM python:3.12-slim


RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*


WORKDIR /VNPT-RAG-Pipeline

COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt


COPY src/ ./src/
COPY predict.py ./
COPY scripts/inference.sh ./inference.sh
RUN chmod +x inference.sh

COPY data/qdrant_storage/ ./data/qdrant_storage/

CMD ["bash", "inference.sh"]
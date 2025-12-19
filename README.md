# 1. Pipeline

```mermaid
---
id: 2180d7b1-ad1f-4ce6-9ab1-1b30d5558cb4
---
graph TD
    Start((Start)) --> Input[/Input JSON Data/]
    

    Input --> Router{Router Decision: VNPT Small}

    Router -- toxic --> ToxicPrompt[Select Refusal Answer VNPT Large]
    
    
    Router -- math --> MathNode[Math Solver: Program of Thought]
    MathNode --> MathPrompt[Prompt Generate Python Code: VNPT Large]
    MathPrompt --> PythonExec{Execute Python REPL}
    PythonExec -- Error/Missing Result --> FixLoop[Feedback Loop: Fix Error & Conclusion]
    FixLoop -- max_tries --> MathPrompt
    PythonExec -- Success --> MathDone[Get Answer from REPL]

    Router -- reading --> ReadingNode[Reading Comprehension: VNPT Large]
    ReadingNode --> Evidence[Extract Evidence & Matching]

    Router -- rag --> RAGNode[RAG: Knowledge Base]
    RAGNode --> Rewrite[Rewriting and Keyword Extraction]
    Rewrite --> Retrieval[Similarity and Keyword Search: Top 20 Docs]
    Retrieval --> Rerank{Reranker: VNPT Small}
    Rerank --> RAGPrompt[Reasoning based on Top 8 Docs: VNPT Large]

    ToxicPrompt & MathDone & Evidence & RAGPrompt --> FinalOutput[Regex: Normalize Answer A-Z]
    FinalOutput --> Export[Write Result to CSV]

    %% Styling
    classDef flow fill:#E1F5FE,stroke:#01579B,stroke-width:2px,color:#000;
    classDef decision fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px,color:#000;
    classDef toxic fill:#FFCDD2,stroke:#C62828,stroke-width:2px,color:#000;
    classDef math fill:#BBDEFB,stroke:#1565C0,stroke-width:2px,color:#000;
    classDef reading fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px,color:#000;
    classDef rag fill:#E1BEE7,stroke:#6A1B9A,stroke-width:2px,color:#000;

    class Start,Input,MainFlow,FinalOutput,Export flow;
    class Router,PythonExec,Rerank decision;
    class ToxicPrompt toxic;
    class MathNode,MathPrompt,FixLoop,MathDone math;
    class ReadingNode,Evidence reading;
    class RAGNode,Rewrite,Retrieval,RAGPrompt rag;
```

## 🛠 Tech Stack

| Component | Implementation |
| :--- | :--- |
| **Orchestration** | **LangChain, LangGraph**|
| **Main LLM (Reasoning)** | **VNPT LLM Large** |
| **Router / Classifier** | **VNPT LLM SMALL** |
| **Vector Database** | **Qdrant** |
| **Retrieval Strategy** | Hybrid (BM25 + Vector) + **LLM-based Reranking** |
| **Code Execution** | **PythonREPL** (LangChain Experimental Sandbox) |
| **Math Solver** | **Program of Thought (PoT)** w/ Self-Correction Loop |
| **Data Processing** | Regex, JSON, Pandas (CSV handling) |
| **Environment** | Python 3.12+, `python-dotenv` |

# 3.  Resource Initialization

Vector Database has been already built and containerized in Docker at: ```/data/qdrant_storage/```
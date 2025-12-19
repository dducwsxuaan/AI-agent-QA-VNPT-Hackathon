# 1. Pipeline

graph TD
    Start((Start)) --> Input[/Input JSON Data/]
    Input --> MainFlow[Main Pipeline: flow method]

    MainFlow --> QueryProc[QueryProcessor: Gemini/Small LLM]
    QueryProc --> Router{Router Decision}

    Router -- toxic --> ToxicNode[Toxic Handler]
    ToxicNode --> ToxicPrompt[Select Refusal Answer]
    
    Router -- math --> MathNode[Math Solver: Program of Thought]
    MathNode --> MathPrompt[Prompt Generate Python Code]
    MathPrompt --> PythonExec{Execute Python REPL}
    PythonExec -- Error/Missing Result --> FixLoop[Feedback Loop: Fix Error & Conclusion]
    FixLoop --> MathPrompt
    PythonExec -- Success --> MathDone[Get Answer from REPL]

    Router -- reading --> ReadingNode[Reading Comprehension]
    ReadingNode --> Evidence[Extract Evidence & Matching]

    Router -- rag --> RAGNode[RAG: Knowledge Base]
    RAGNode --> Retrieval[Similarity Search: Top 10 Docs]
    Retrieval --> Rerank{Reranker: Small LLM}
    Rerank --> RAGPrompt[Reasoning based on Top 3 Docs]

    ToxicPrompt & MathDone & Evidence & RAGPrompt --> FinalOutput[Regex: Normalize Answer A-D]
    FinalOutput --> Export[Write Result to CSV]

# 3.  Resource Initialization

Vector Database has been already built and containerized in Docker at: ```/data/qdrant_storage/```
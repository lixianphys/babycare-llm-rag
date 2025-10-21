## BabyCare Chatbot Architecture

This document outlines the high-level architecture of the Integrated Baby Care Chatbot with RAG.

### Components Overview
- **IntegratedBabyCareChatbot**: Orchestrates the chat flow using LangGraph `StateGraph` and routes between clarification, RAG, and fallback.
- **BabyCareRAGSystem**: Generates responses using retrieved context; integrates optional `LangSmithMonitor` for usage/cost tracking.
- **Vector Store**: `SimpleBabyCareVectorStore` (in-memory) or `BabyCareVectorStore` (persistent) selected via `config.use_memory_vector_store`.
- **Knowledge Base**: Ingested from PDFs and other files, or seeded via `create_sample_baby_care_documents()`.

### High-level Architecture
```mermaid
flowchart LR
  subgraph Client
    U[User]
  end

  subgraph App
    A1[Analyze Query]
    A2[Retrieve Context]
    A3[Generate RAG Response]
    A4[Request Clarification]
    A5[Fallback Response]
    G[StateGraph]
  end

  subgraph RAG
    RS[BabyCareRAGSystem]
    MON[LangSmithMonitor]
  end

  subgraph KB
    VS1[Simple Vector Store]
    VS2[Persistent Vector Store]
    DOCS[(Documents)]
  end

  U --> G
  G --> A1
  A1 --> A4
  A1 --> A5
  A1 --> A2
  A2 --> A3
  A3 --> U
  A2 --> VS1
  A2 --> VS2
  A3 --> RS
  RS --> MON
  DOCS --> VS1
  DOCS --> VS2
```

### Request Flow (Sequence)
```mermaid
sequenceDiagram
  participant User
  participant Bot as IntegratedBabyCareChatbot
  participant Graph as StateGraph(BabyCareState)
  participant VS as VectorStore (Simple/Persistent)
  participant RAG as BabyCareRAGSystem
  participant Mon as LangSmithMonitor

  User->>Bot: chat(user_input) / stream_chat(...)
  Bot->>Graph: invoke(initial_state)
  Graph->>Bot: run analyze_query
  alt Needs clarification
    Bot->>Graph: request_clarification
    Graph-->>User: AIMessage(clarification)
  else Should use RAG
    Bot->>VS: search_documents(query, k=top_k)
    Bot->>RAG: generate_response(query, chat_history)
    RAG-->>Bot: response
    Bot-->>User: AIMessage(response)
    opt Monitoring enabled
      RAG->>Mon: log usage/cost
    end
  else Fallback
    Bot-->>User: AIMessage(fallback)
  end
```

### Notes
- Initialization optionally seeds the KB when empty via `create_sample_baby_care_documents()`.
- `get_knowledge_base_info`, `get_rag_keywords`, and `search_knowledge_base` expose useful diagnostics and search endpoints.
- Ingestion helpers: `add_documents`, `add_pdf_folder`, `add_documents_from_folder`.



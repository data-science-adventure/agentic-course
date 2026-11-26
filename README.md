
# 🦜 Agentic AI Course: LangChain & LangGraph Masterclass

A hands-on, module-based course for building production-ready AI agents and retrieval pipelines using **LangChain** and **LangGraph**, powered by modern tool-calling architectures, vector retrieval, and local/cloud LLMs.

## 📂 Project Architecture

The course is organized into structured modules within the `src/` directory. Each lesson includes CLI entry points managed via `uv`.

```shell
agentic-course
├── agent_output
│   ├── multi_tool_agent_messages.json
│   └── shopping_state_graph_messages.json
├── index.html
├── llms.excalidraw
├── pyproject.toml
├── README.md
├── src
│   ├── agentic_course
│   │   ├── main.py
│   │   ├── module_01_basics
│   │   │   ├── llm_intro.py
│   │   ├── module_02_tools
│   │   │   ├── multi_tool_agent.py
│   │   │   └── simple_tools.py
│   │   ├── module_03_langgraph
│   │   │   └── shopping_state_graph.py
│   │   ├── module_04_rag
│   │   │   ├── create_pinecone_index.py
│   │   │   ├── load_rag_pdf.py
│   │   │   └── query_rag_pdf.py
│   │   └── utils
│   │       └── save_message_to_json.py
└── uv.lock
```

## 🛠️ Core Concepts Covered

### Module 01: Foundations & LCEL

- **LangChain Expression Language (LCEL)**: Building modular chains using `PromptTemplate | LLM | OutputParser`.
- **Local Hugging Face Pipelines**: Running lightweight models like `gpt2` without external API dependencies.
- **LangSmith Telemetry**: Automatic tracing of chain execution via environment variables.

### Module 02: Tool Integration & Modern Agents

- **`@tool` Decorator**: Converting standard Python functions into LLM-understandable tools.
- **`create_agent` Factory**: Assembling tool-calling agents backed by standard graph execution loops.
- **Docstring Engineering**: Writing type-annotated docstrings that LLMs use to infer _when_ and _how_ to invoke custom tools.
- **Compound Queries**: Multi-tool execution in a single reasoning pass.

### Module 03: LangGraph & Orchestration Foundations

- **Model Swapping & Tool Binding**: Using `init_chat_model` with `ChatGroq` (`groq:...`) and `.bind_tools(...)` for tool-driven workflows.
- **Cyclic Control Flow & State**: Comparing iterative tool-calling loops against explicit **LangGraph** **`StateGraph`** orchestration, **node**s, **conditional_edge**s, and **checkpointer**s.

### Module 04: Retrieval-Augmented Generation (RAG) & Pinecone

- **Index Provisioning & Dimension Alignment**: Creating a **Pinecone** serverless index with explicit `dimension` and `metric` (`cosine`) matched to your embedding model (`384` dims for `sentence-transformers/all-MiniLM-L6-v2`, or `1536` dims for `text-embedding-3-small`).
- **Document Loading & Chunking**: Loading PDFs with `PyPDFLoader` and splitting into overlapping chunks with `RecursiveCharacterTextSplitter` before upserting via `PineconeVectorStore`.
- **Grounded Generation**: Retrieving relevant chunks with `.as_retriever(...)` and generating answers with `ChatGroq` conditioned on retrieved context.

### Utilities: State Inspection

- **`save_messages_to_json`**: Serializing runtime message histories (`HumanMessage`, `AIMessage`, `ToolMessage`) into JSON files under `agent_output/` for debugging.

## 🚀 Getting Started with `uv`

### 1. Environment Setup

Clone the repository and sync project dependencies using `uv`:

```shell
uv sync
```

Set up your `.env` file in the root directory:

```shell
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
INDEX_NAME=your_pinecone_index_name
```

## 🏃 Running Lessons

You can run lessons using either the **interactive menu** or direct **CLI script commands**.

### Option A: Interactive Menu (Recommended) 🎨

Run the dynamic lesson selector:

```shell
uv run show-modules
```

### Option B: Direct CLI Execution ⚡

Execute specific lesson scripts directly:

```shell
# Run Module 01: Hello World LCEL Demo
uv run m1-intro

# Run Module 02: Custom Tools Agent
uv run m2-tools

# Run Module 02: Multi-Tool Agent Execution
uv run m2-multi-tool

# Run Module 03: Shopping Workflow
uv run m3-shopping

# Run Module 04: Create / Reset Pinecone Index
uv run m4-create-index

# Run Module 04: Load and Split PDF into Pinecone
uv run m4-load-pdf

# Run Module 04: Query Indexed PDF with ChatGroq
uv run m4-query-pdf
```

## 💡 Best Practices for Custom Tools & RAG

1. **Explicit Docstrings & Type Hints**: Clear docstrings and explicit type annotations (e.g., `product: str`, `k: int = 4`) give models and callers unambiguous contracts.
2. **Embedding Dimension Lockstep**: Always match the Pinecone index `dimension` to your embedding model's output size to avoid `400` dimension errors.
3. **Structured Returns & Grounded Context**: Return clear text/JSON from tools and format retrieved passages with source/page metadata so generation stays grounded.
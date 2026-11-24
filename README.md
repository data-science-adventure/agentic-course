# 🦜 Agentic AI Course: LangChain & LangGraph Masterclass

A hands-on, module-based course for building production-ready AI agents using **LangChain** and **LangGraph**, powered by modern tool-calling architectures and local/cloud LLMs.

## 📂 Project Architecture

The course is organized into structured modules within the `src/` directory. Each lesson includes CLI entry points managed via `uv`.

```shell
agentic-course
├── README.md
├── pyproject.toml
├── uv.lock
├── inspect.json                  # Message history inspection output
├── src
│   └── agentic_course
│       ├── main.py               # Interactive CLI Menu Runner
│       ├── module_01_basics      # Module 1: Core LangChain & LCEL
│       │   └── llm_intro.py
│       ├── module_02_tools       # Module 2: Tools & ReAct Agents
│       │   ├── simple_tools.py
│       │   └── multi_tool_agent.py
│       └── utils                 # Shared Helper Utilities
│           └── save_message_to_json.py
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

### Utilities: State Inspection

- **`save_messages_to_json`**: Serializing runtime message histories (`HumanMessage`, `AIMessage`, `ToolMessage`) into formatted JSON for visual debugging.

## 🚀 Getting Started with `uv`

### 1. Environment Setup

Clone the repository and sync project dependencies using `uv`:

```bash
uv sync
```

Set up your `.env` file in the root directory:

```shell
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
GROQ_API_KEY=your_groq_key
```

## 🏃 Running Lessons

You can run lessons using either the **interactive menu** or direct **CLI script commands**.
### Option A: Interactive Menu (Recommended) 🎨

Run the dynamic lesson selector:

```bash
uv run run-main
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
```

## 💡 Best Practices for Custom Tools

When writing tools for AI agents:

1. **Explicit Docstrings**: The docstring forms the tool schema supplied to the LLM. Clear descriptions prevent invalid parameter calls.
2. **Type Hints**: Always add explicit typing (e.g., `location: str`) so the function arguments parse correctly.
3. **Structured Returns**: Return clear text or JSON strings that the model can parse and incorporate into its final response.
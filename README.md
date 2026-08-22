# Module 2: Modern Agent Creation & Tool Integration (`create_agent`)

In this module, you will learn how to build a modern tool-calling agent using **LangChain's standard agent architecture** powered by **Groq Cloud**.

---

## 🛠️ 1. Core Concepts

* **`@tool` Decorator**: Converts standard Python functions into tools usable by LLMs.
* **`create_agent`**: The modern factory function that binds the language model, custom tools, and system prompt into an execution graph.
* **Message-Based Invocation**: Modern LangChain agents receive structured message lists (e.g., `{"messages": [{"role": "user", "content": "..."}]}`) and append model and tool responses into the chat history.

---

## 💡 2. The Importance of Tool Documentation & Descriptions

When creating custom tools with the `@tool` decorator, **the Python docstring and type hints act as the tool's primary specification for the LLM**. 

* **How LLMs see your tools**: The function name, parameter type annotations, and docstring are serialized into a JSON Schema provided directly to the model.
* **Why docstrings matter**: The model reads the docstring to determine **if** and **when** it should invoke a tool. If a docstring is missing, vague, or ambiguous, the model may fail to call the tool or supply invalid arguments.
* **Best Practices**:
  1. Write explicit, concise docstrings that describe what the tool does and what input formats it expects.
  2. Always include clear type hints for function arguments (e.g., `location: str`).

---

## 🏃 3. Execution

Run the updated main script via `uv`:

```bash
uv run main.py
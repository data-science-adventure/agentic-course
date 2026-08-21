# Agentic with LangChain Ecosystem - Class 1 Demo

Welcome to the **Agentic with LangChain Ecosystem** course project template. This repository contains a "Hello World" demonstration of building LLM chains using the LangChain Expression Language (LCEL), running local Hugging Face models, and configuring observational tracing with **LangSmith**.


## 🛠️ Prerequisites

Before getting started, make sure you have the following installed on your system:

1. **Python**: Version `3.12` or higher (as configured in `uv.lock` / project specifications).
2. **uv**: An extremely fast Python package and project manager.
   - Install `uv` via official script:
     ```bash
     # macOS / Linux
     curl -LsSf https://astral.sh/uv/install.sh | sh

     # Windows (PowerShell)
     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
3. **LangSmith Account & API Key**:
   - Sign up/log in at [LangSmith](https://smith.langchain.com/).
   - Generate an API key from your account settings.

---

## ⚙️ Environment Configuration

Create a `.env` file in the root directory of your project to set up LangSmith tracing and project variables:

```shell
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<YOUR_API_KEY>
LANGSMITH_PROJECT=<YOUR_LANG_SMITH_PROJECT_NAME>
```

> **Note**: Replace `<YOUR_API_KEY>` and `<YOUR_LANG_SMITH_PROJECT_NAME>` with your actual LangSmith credentials and desired project name.


## 🚀 Getting Started & Installation

Using `uv`, dependency management and virtual environment creation are fast and seamless.

### 1. Sync Dependencies & Setup Virtual Environment

Run the following command in the project root to install all required dependencies (including `langchain`, `langchain-huggingface`, `rich`, `torch`, `transformers`, and `python-dotenv`):

```bash
uv sync
```

This will automatically create a virtual environment (`.venv`) and install exact locked versions.

## 🏃 Running the Project

To execute the demo script using `uv`:

```bash
uv run python main.py
```

### What happens when you run it:
1. **Environment Setup**: `.env` variables are loaded to initialize LangSmith tracing.
2. **Model Loading**: The script initializes a local Hugging Face model pipeline (`gpt2` for text generation).
3. **Prompt & Chain**: Builds an LCEL pipeline combining a `PromptTemplate`, the `HuggingFacePipeline`, and a `StrOutputParser`.
4. **Execution**: Invokes the chain with a sample query (`"How are you?"`) and renders rich terminal output using `rich`.
5. **Observability**: Execution traces (prompts, latency, inputs, outputs) are automatically sent to your **LangSmith Dashboard**.


## Viewing Traces in LangSmith

After running the script:
1. Open your [LangSmith Dashboard](https://smith.langchain.com/).
2. Select the project name defined in your `LANGSMITH_PROJECT` environment variable.
3. Review the complete execution graph, run times, prompt inputs, and model outputs.
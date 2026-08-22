# Module 1: Building LLM Chains with Groq & LangChain Expression Language (LCEL)

In this exercise, you will learn how to transition from local models or paid services to using high-performance open-weight models powered by **Groq Cloud**. You will set up your environment, configure API access, understand Groq's available model ecosystem, and build an LCEL chain.

---

## 🛠️ 1. Installing Required Dependencies

We use `uv` for package management. Run the following command in your terminal to add `langchain-groq` to your virtual environment:

```bash
uv add langchain-groq
```

This updates your `pyproject.toml` and `uv.lock` files automatically.

---

## 🔑 2. Groq Cloud Account Registration & API Key

Groq offers a generous free tier with high inference speeds (~1000 tokens/sec).

### Step-by-Step Registration:
1. Navigate to the **[Groq Console](https://console.groq.com)**.
2. Sign up using your Google, GitHub, or email account.
3. In the left sidebar, navigate to **API Keys**.
4. Click **Create API Key**.
5. Copy the generated key (starts with `gsk_...`). *Keep this key private and never commit it to source control.*

---

## ⚙️ 3. Environment Variables Configuration

In the root of your project, open or create your `.env` file and append your Groq API key along with your LangSmith tracing variables:

```env
# LangSmith Observability
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<YOUR_LANGSMITH_API_KEY>
LANGSMITH_PROJECT=<YOUR_LANGSMITH_PROJECT_NAME>

# Groq API Credentials
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

---

## 📊 4. Groq Model Ecosystem Overview

Groq supports a diverse range of specialized models, compound systems, audio processing, and security classifiers:

| Model ID | Primary Purpose | Type of LLM | Parameters / Architecture | Context Window | Key Characteristics & Best Use Cases |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`qwen/qwen3.6-27b`** | Advanced Coding & Reasoning | Dense LLM (Extended Thinking / Reasoning) | 27 Billion | 262,144 tokens | Features hybrid multimodal capabilities and a thinking mode. Excellent for repository-level coding, structured outputs (PlantUML, JSON), and multi-step agentic workflows. |
| **`openai/gpt-oss-120b`** | Complex Problem-Solving | Mixture of Experts (MoE LLM) | 117 Billion (5.1B active) | 131,072 tokens | High-capacity model built for deep chain-of-thought, complex reasoning, and native tool calling where output quality is prioritized over speed. |
| **`openai/gpt-oss-20b`** | General Purpose & High-Speed Inference | Mixture of Experts (MoE LLM) | 20 Billion | 131,072 tokens | Ultra-fast execution (~1000 tokens/sec on Groq). Ideal for general dialogue, rapid prototyping, and real-time agentic interactions. |
| **`allam-2-7b`** | Bilingual General Text | Dense LLM | 7 Billion | 8,192 tokens | Specialized for high-quality Arabic and English task performance and text generation. |
| **`groq/compound`** | Server-side Agentic Workflows | Compound AI System | Multi-model orchestration (GPT-OSS 120B / Llama) | 131,072 tokens | An autonomous system that executes server-side tool calls (e.g., multiple web searches, code execution) in a single request. |
| **`groq/compound-mini`** | Fast Single-Tool Workflows | Compound AI System | Lightweight Multi-model orchestration | 131,072 tokens | Lightweight version of `groq/compound` optimized for execution requiring a single tool call per user query. |
| **`canopylabs/orpheus-v1-english`** | Text-To-Speech (TTS) | Audio Synthesis Model | Expressive Audio Architecture | 4,000 characters | Generates high-quality English audio. Accepts inline style bracket directions (e.g., `[cheerful]`, `[whisper]`). |
| **`canopylabs/orpheus-arabic-saudi`** | Dialect Text-To-Speech | Audio Synthesis Model | Expressive Audio Architecture | 4,000 characters | Generates natural, authentic Saudi dialect spoken audio from input text. |
| **`whisper-large-v3`** | Speech-To-Text (STT) | Audio Transcription Model | Sequence-to-Sequence Audio Encoder | 25 MB max file | Converts multilingual spoken audio files directly into structured transcripts or translations. |
| **`whisper-large-v3-turbo`** | Low-Latency Speech Transcriptions | Audio Transcription Model | Pruned Audio Encoder | 25 MB max file | Optimized, low-cost variant of Whisper Large v3 designed for real-time live transcriptions. |
| **`meta-llama/llama-prompt-guard-2-86m`** | Input Guardrail & Moderation | Moderation Classifier | 86 Million | 512 tokens | Classifies incoming user prompts to detect prompt injections, jailbreaks, and harmful input vectors before reaching the main LLM. |
| **`meta-llama/llama-prompt-guard-2-22m`** | Ultra-fast Input Security | Moderation Classifier | 22 Million | 512 tokens | Ultra-lightweight prompt security classifier optimized for near-zero latency checks. |
| **`openai/gpt-oss-safeguard-20b`** | Output Alignment & Safety | Safety Classifier | 20 Billion | 131,072 tokens | Evaluates model responses against target safety guidelines to filter unsafe or policy-violating text. |

---

## 🚀 5. Implementation Code (`src/agents/groq_agent.py`)

Create the agent script in `src/agents/groq_agent.py`:

```python
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from rich.console import Console
from rich.panel import Panel

# Load environment variables (.env)
load_dotenv()

console = Console()

def main():
    console.print(
        Panel.fit(
            "[bold green]LangChain Groq Demo[/bold green]\n"
            "[dim]Demonstrating ChatGroq integration with LCEL & LangSmith tracing[/dim]",
            border_style="green",
        )
    )

    if not os.getenv("GROQ_API_KEY"):
        console.print(
            "[bold red]Error:[/bold red] GROQ_API_KEY environment variable is missing in your .env file."
        )
        return

    # Initialize Groq LLM (e.g., Qwen reasoning/coding model)
    model = ChatGroq(
        model="qwen/qwen3.6-27b",
        temperature=0.7,
    )

    prompt = PromptTemplate.from_template(
        "You are a helpful and polite AI assistant. Answer the user's query: {query}"
    )

    # Build LCEL chain
    chain = prompt | model | StrOutputParser()

    query = "How are you?"
    console.print(f"\n[bold yellow]User Query:[/bold yellow] {query}\n")

    with console.status("[bold cyan]Generating response from Groq...[/bold cyan]"):
        response = chain.invoke({"query": query})

    console.print(
        Panel(
            response,
            title="[bold green]Groq Response[/bold green]",
            border_style="green",
        )
    )

if __name__ == "__main__":
    main()
```

---

## 🏃 6. Running the Agent

Run the script using Python module execution syntax with `uv`:

```bash
uv run python -m src.agents.groq_agent
```
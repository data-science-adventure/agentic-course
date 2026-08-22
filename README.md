# Module 3: Web Search Tool Integration with Tavily & Groq

In this module, you will learn how to integrate real-time external web search into your agents using **`langchain-tavily`** and **`ChatGroq`**.

---

## 🛠️ 1. Core Concepts

* **`TavilySearch`**: Tavily partner tool (`langchain-tavily`) designed specifically for LLMs to fetch web search results cleanly.
* **Tool Loop Execution**: How `create_agent` coordinates model reasoning, tool execution, and final text synthesis.
* **Token Budgeting**: Managing payload sizes with `max_results=1` to remain within provider API rate limits.

---

## 🔑 2. Setup Dependencies & Environment

Install package dependencies using `uv`:

```bash
uv add langchain-tavily
```

Obtain your API keys:
* **Groq API Key**: Get it from the [Groq Console](https://console.groq.com/).
* **Tavily API Key**: Register and obtain a free API key at [Tavily AI Registry](https://tavily.com/).

Add your API credentials to your `.env` file:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
TAVILY_API_KEY=tvly-your_tavily_api_key_here
```

---


## 🏃 3. Execution

Run the script via `uv`:

```bash
uv run main.py
```
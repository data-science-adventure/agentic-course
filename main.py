from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from rich.console import Console
from rich.panel import Panel
from datetime import datetime

load_dotenv()
console = Console()

# 1. Define Tools using @tool decorator
@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for a given location."""
    location_lower = location.lower()
    if "tokyo" in location_lower:
        return "18°C, Light Rain"
    elif "madrid" in location_lower:
        return "26°C, Sunny"
    else:
        return "22°C, Partly Cloudy"


@tool
def get_time() -> str:
    """Gets the current local date and time.

    Returns:
        str: A formatted string containing the current date and time (YYYY-MM-DD HH:MM:SS).
    """
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def main():

    # Tools array
    tools = [get_current_weather, get_time]

    # Initialize Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    # Modern LangChain agent initialization
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful AI assistant equipped with tools. Use your available tools when necessary to answer questions.",
    )

    query = "What is the weather in Tokyo right now, and how many words are in the sentence 'Artificial Intelligence is awesome'? and Give me the time"
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    
    # The final answer lives in the last message
    final_output = response["messages"][-1].content
    console.print(final_output)

if __name__ == "__main__":
    main()

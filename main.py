from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from rich.console import Console
from rich.panel import Panel

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
def calculate_word_length(text: str) -> str:
    """Calculate the number of characters and words in a given text string."""
    word_count = len(text.split())
    char_count = len(text)
    return f"Word count: {word_count}, Character count: {char_count}"


def main():

    # Tools array
    tools = [get_current_weather, calculate_word_length]

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

    query = "What is the weather in Tokyo right now, and how many words are in the sentence 'Artificial Intelligence is awesome'?"
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    
    # The final answer lives in the last message
    final_output = response["messages"][-1].content
    console.print(final_output)

if __name__ == "__main__":
    main()

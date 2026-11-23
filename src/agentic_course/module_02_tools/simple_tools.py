"""Module 02: Lesson 2.1 - Custom Tools and Tool-Calling Agents.

This script demonstrates:
1. Defining custom tools using the @tool decorator and typing docstrings.
2. Initializing a model with native tool-calling support via ChatGroq.
3. Using `create_agent` to assemble a ReAct agent equipped with multiple tools.
4. Executing multi-step reasoning queries that trigger tool calls.
"""

from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()


@tool
def get_time() -> str:
    """Gets the current local date and time.

    Returns:
        str: A formatted string containing current date and time (YYYY-MM-DD HH:MM:SS).
    """
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for a given location.

    Args:
        location (str): The name of the city/location to check.

    Returns:
        str: Weather summary including temperature and condition.
    """
    location_lower = location.lower()
    if "tokyo" in location_lower:
        return "18°C, Light Rain"
    elif "madrid" in location_lower:
        return "26°C, Sunny"
    else:
        return "22°C, Partly Cloudy"


def main():
    """Main execution entrypoint for CLI execution."""
    console.rule("[bold cyan]🛠️ Module 02: Simple Tools Agent[/bold cyan]")
    console.print(
        "\n[bold yellow]🚀 Initializing ChatGroq & Registering Tools...[/bold yellow]\n"
    )

    tools = [get_current_weather, get_time]

    # Initialize ChatGroq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    # Construct the agent runtime using standard create_agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful AI assistant equipped with tools. "
            "Use your available tools when necessary to answer questions."
        ),
    )

    query = (
        "What is the weather in Tokyo right now, and how many words are in the "
        "sentence 'Artificial Intelligence is awesome'? Also give me the current time."
    )

    console.print(Panel(f"[italic]{query}[/italic]", title="💬 User Query", border_style="cyan"))
    console.print("\n[bold magenta]Processing agent execution loop...[/bold magenta]\n")

    # Run the agent graph
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})

    # Extract final agent response
    final_message = response["messages"][-1].content

    console.print("[bold green]--- Agent Output ---[/bold green]")
    console.print(f"[bright_white]{final_message.strip()}[/bright_white]")
    console.print("[bold green]----------------------[/bold green]")
    console.print("\n[bold cyan]✅ Execution completed successfully![/bold cyan]\n")


if __name__ == "__main__":
    main()
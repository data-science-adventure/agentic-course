"""Module 02: Lesson 2.2 - Multi-Tool Agent Execution.

This script demonstrates:
1. Registering multiple custom tools (@tool) within a single agent setup.
2. Initializing ChatGroq as the inference backbone.
3. Executing a complex, multi-intent prompt using `create_agent`.
4. Inspecting how the agent orchestrates tool selection and text generation.
"""

from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from rich.console import Console
from rich.panel import Panel
from agentic_course.utils.save_message_to_json import save_messages_to_json

load_dotenv()
console = Console()


@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for a given location.

    Args:
        location (str): Name of the city to query.

    Returns:
        str: Weather conditions string.
    """
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
        str: Formatted timestamp string (YYYY-MM-DD HH:MM:SS).
    """
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def main():
    """Main execution entrypoint for CLI runner."""
    console.rule("[bold cyan]🤖 Module 02: Multi-Tool Agent Execution[/bold cyan]")
    console.print(
        "\n[bold yellow]🚀 Initializing ChatGroq & Registering Tools...[/bold yellow]\n"
    )

    # 1. Tool selection
    tools = [get_current_weather, get_time]

    # 2. Model setup
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    # 3. Compile the Agent Graph
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
        "sentence 'Artificial Intelligence is awesome'? and Give me the time"
    )

    console.print(
        Panel(f"[italic]{query}[/italic]", title="💬 Compound Query", border_style="cyan")
    )
    console.print("\n[bold magenta]Processing agent execution loop...[/bold magenta]\n")

    # 4. Invoke agent workflow
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})

    # Extract final text output from state messages history
    final_output = response["messages"][-1].content
    save_messages_to_json(response["messages"])

    console.print("[bold green]--- Agent Final Output ---[/bold green]")
    console.print(f"[bright_white]{final_output.strip()}[/bright_white]")
    console.print("[bold green]---------------------------[/bold green]")
    console.print(
        "\n[bold cyan]✅ Execution finished! Check LangSmith for intermediate tool traces.[/bold cyan]\n"
    )


if __name__ == "__main__":
    main()
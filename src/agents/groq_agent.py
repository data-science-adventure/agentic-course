import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

def main():
    if not os.getenv("GROQ_API_KEY"):
        console.print("[bold red]Error:[/bold red] GROQ_API_KEY is missing in .env")
        return

    # Initialize model (runs on Groq's fast hardware)
    model = ChatGroq(
        model="qwen/qwen3.6-27b",
        temperature=0.7,
    )

    prompt = PromptTemplate.from_template(
        "You are a helpful AI assistant. Answer the query: {query}"
    )

    # LCEL Chain
    chain = prompt | model | StrOutputParser()

    query = "How are you?"
    console.print(f"[bold yellow]User Query:[/bold yellow] {query}\n")

    with console.status("[bold cyan]Generating response via Groq...[/bold cyan]"):
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
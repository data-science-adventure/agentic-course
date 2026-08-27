from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()

def main():

    # Initialize model (runs on Groq's fast hardware)
    llm = ChatGroq(
        model="qwen/qwen3.6-27b",
        temperature=0.7,
    )

    prompt = PromptTemplate.from_template(
        "You are a helpful AI assistant. Answer the query: {query}"
    )

    # LCEL Chain
    chain = prompt | llm | StrOutputParser()

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
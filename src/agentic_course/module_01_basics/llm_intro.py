"""Module 01: Lesson 1.1 - Hello World with LCEL & Local Hugging Face Models.

This script demonstrates:
1. Loading environment variables for LangSmith telemetry.
2. Initializing a local open-source LLM using HuggingFacePipeline.
3. Constructing an LCEL chain (Prompt | Model | OutputParser).
4. Running simple inference with formatted terminal logging.
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from rich.console import Console

console = Console()


def build_chain():
    """Initializes model components and constructs the LCEL pipeline."""
    # Initialize local Hugging Face model
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 50,
            "temperature": 0.7,
            "pad_token_id": 50256,
        },
    )

    # Create Prompt Template
    prompt = PromptTemplate.from_template(
        "Question: {pregunta}\nAnswer in simple terms:"
    )

    # Build Chain using LangChain Expression Language (LCEL)
    chain = prompt | llm | StrOutputParser()
    return chain


def main():
    """Main execution entrypoint for CLI execution."""
    # Load environment variables to enable LangSmith tracing
    load_dotenv()

    console.rule("[bold cyan]🤖 AI Agent - Class 1 Demo[/bold cyan]")
    console.print(
        "\n[bold yellow]🚀 Initializing LangSmith & Local Model...[/bold yellow]\n"
    )

    chain = build_chain()
    console.print("[bold green]✓ Model loaded successfully![/bold green]\n")

    # Run execution
    pregunta_usuario = "How are you?"

    console.print(
        f"[bold white]Question:[/bold white] [italic]{pregunta_usuario}[/italic]\n"
    )
    console.print(
        "[bold magenta]Processing response with LangChain...[/bold magenta]\n"
    )

    respuesta = chain.invoke({"pregunta": pregunta_usuario})

    # Output results
    console.print("[bold green]--- Response Generated ---[/bold green]")
    console.print(f"[bright_white]{respuesta.strip()}[/bright_white]")
    console.print("[bold green]--------------------------[/bold green]")

    console.print(
        "\n[bold cyan]✅ Execution finished! Check your LangSmith dashboard for the trace.[/bold cyan]\n"
    )


if __name__ == "__main__":
    main()
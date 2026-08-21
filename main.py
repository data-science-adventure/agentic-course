import os
from dotenv import load_dotenv
from rich.console import Console
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

console = Console()

def main():
    # 1. Load environment variables to enable LangSmith tracing
    load_dotenv()

    console.rule("[bold cyan]🤖 AI Agent - Class 1 Demo[/bold cyan]")
    console.print("\n[bold yellow]🚀 Initializing LangSmith & Local Model...[/bold yellow]\n")

    # 2. Initialize Hugging Face model
    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 50,
            "temperature": 0.7,
            "pad_token_id": 50256,
        },
    )

    console.print("[bold green]✓ Model loaded successfully![/bold green]\n")

    # 3. Create Prompt Template
    prompt = PromptTemplate.from_template(
        "Question: {pregunta}\nAnswer in simple terms:"
    )

    # 4. Build Chain using LCEL
    chain = prompt | llm | StrOutputParser()

    # 5. Run execution
    pregunta_usuario = "How are you?"
    
    console.print(f"[bold white]Question:[/bold white] [italic]{pregunta_usuario}[/italic]\n")
    console.print("[bold magenta]Processing response with LangChain...[/bold magenta]\n")

    respuesta = chain.invoke({"pregunta": pregunta_usuario})

    # Output results
    console.print("[bold green]--- Response Generated ---[/bold green]")
    console.print(f"[bright_white]{respuesta.strip()}[/bright_white]")
    console.print("[bold green]--------------------------[/bold green]")
    
    console.print("\n[bold cyan]✅ Execution finished! Check your LangSmith dashboard for the trace.[/bold cyan]\n")

if __name__ == "__main__":
    main()
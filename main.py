from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from rich.console import Console

load_dotenv()
console = Console()

# Initialize Groq LLM (gpt-oss-20b executes tool loops reliably)
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# Limit results to 1 to keep context within token limits
tools = [TavilySearch(max_results=1)]

agent = create_agent(
    model=llm, 
    tools=tools,
    system_prompt="You are a helpful AI assistant equipped with web search. Always synthesize search results into a concise final response."
)

def main():
    console.print("[bold green]Starting Module 3: Tavily Web Search Agent...[/bold green]")
    
    query = "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
    
    # Pass input as standard string key
    result = agent.invoke({"messages": [("user", query)]})
    
    # Extract final message content
    final_output = result["messages"][-1].content
    
    console.print("\n[bold yellow]Agent Response:[/bold yellow]")
    console.print(final_output)


if __name__ == "__main__":
    main()
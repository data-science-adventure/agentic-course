from typing import List
from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for an individual source cited by the agent."""

    url: str = Field(description="The web URL of the cited source.")


class AgentResponse(BaseModel):
    """Schema for final structured answer and its citations."""

    answer: str = Field(description="The comprehensive text answer.")
    sources: List[Source] = Field(
        default_factory=list,
        description="List of web sources used to answer the query.",
    )


from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_groq import ChatGroq

load_dotenv()

# Initialize LLM
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# Bind the Pydantic schema using ToolStrategy
agent = create_agent(
    model=llm,
    tools=[],  # Add external tools here if needed
    response_format=ToolStrategy(AgentResponse),
    system_prompt="Provide concise single-paragraph answers without line breaks or markdown formatting in your tool arguments.",
)

# Execute the agent
result = agent.invoke(
    {
        "messages": [
            ("user", "What is LangChain? Provide the answer and mock source URL.")
        ]
    }
)

# Access typed output via "structured_response"
structured_data = result["structured_response"]
print(f"Answer: {structured_data.answer}")
print(f"Sources: {structured_data.sources}")

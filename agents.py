from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
import os

# Configuraciones de API
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

def create_tavily_agent(model_id, temperature=0.7):
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    llm = ChatGroq(model=model_id, temperature=temperature)
    search = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search)
    agent_chain = initialize_agent(
        [tavily_tool],
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent_chain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool
import logging

logger = logging.getLogger(__name__)

def get_web_search_tool():
    """
    Creates a Web Search tool using DuckDuckGo (Free, no API key).
    This allows the Agent to find public information not present in the internal documents.
    """
    try:
        # Wrapper provides more control (e.g. region, time) if needed
        # Explicitly setting backend="duckduckgo" as it is the correct identifier for the text search API
        wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", max_results=5, backend="duckduckgo")
        search = DuckDuckGoSearchRun(api_wrapper=wrapper)
        
        return Tool(
            name="web_search",
            func=search.run,
            description="""
            Useful for searching the public internet for current events, competitor analysis, 
            market comparables, or general facts that are likely not in the internal 
            knowledge base. Use this when the user asks about 'latest', 'news', 'competitors', 
            or specific public data points.
            """
        )
    except Exception as e:
        logger.error(f"Failed to initialize DuckDuckGo search: {e}")
        # Fallback to a dummy tool so the agent doesn't crash
        def dummy_search(query):
            return "Web search is currently unavailable."
            
        return Tool(
            name="web_search",
            func=dummy_search,
            description="Web search is currently unavailable."
        )

from langchain_core.tools import tool

@tool
def web_search(query: str, max_results: int = 6) -> str:
    """
    Searches the web using DuckDuckGo to find current information.
    Use this tool to find market data, news, industry trends, competitor info,
    financial data, or any real-world knowledge that is not in the provided dataset.
    
    Args:
        query: The search query string.
        max_results: Number of results to return (default 6, max 10).
    
    Returns a formatted string of search results with titles, URLs, and snippets.
    """
    try:
        from duckduckgo_search import DDGS
        
        max_results = min(int(max_results), 10)
        results = []
        
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))
        
        if not search_results:
            return f"No results found for query: '{query}'"
        
        formatted = [f"## Web Search Results for: '{query}'\n"]
        for i, r in enumerate(search_results, 1):
            title = r.get("title", "No Title")
            href  = r.get("href", "")
            body  = r.get("body", "No snippet available.")
            formatted.append(
                f"**Result {i}: {title}**\n"
                f"URL: {href}\n"
                f"Summary: {body}\n"
            )
        
        return "\n".join(formatted)
    
    except ImportError:
        return (
            "Error: 'duckduckgo-search' package is not installed. "
            "Run: pip install duckduckgo-search"
        )
    except Exception as e:
        return f"Web search failed: {str(e)}"

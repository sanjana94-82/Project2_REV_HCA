# app/tools.py

import os
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

# Optional: load key from environment
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in environment variables.")

def get_tavily_tool():
    return TavilySearchResults(k=3)

# Optional: SQL Tool Example
def get_sql_tool():
    db = SQLDatabase.from_uri("sqlite:///data/medical.sqlite")
    return SQLDatabaseToolkit(db=db)


import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

def get_sql_tool():
    db = SQLDatabase.from_uri("sqlite:///data/medical.sqlite")
    return SQLDatabaseToolkit(db=db)
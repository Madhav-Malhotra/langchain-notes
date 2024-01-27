# Usual libraries
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain.llms.huggingface_hub import HuggingFaceHub

# SQL specific libraries
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# Load environmental variable
load_dotenv(".env")

# Setup LLM (though this one isn't powerful enough to work well with SQL)
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={
    'temperature': 0.01, 'max_length': 64
})

# URI doesn't need to be local file. Could be a remote DB.
db = SQLDatabase.from_uri("sqlite:///l12.db")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# Run agent
agent_executor.run("Describe the largest table in the database.")
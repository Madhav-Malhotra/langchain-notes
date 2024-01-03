from dotenv import load_dotenv
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.agents import tool, load_tools, initialize_agent, AgentType

# Load environmental variable
load_dotenv(".env")

def create_agent():
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={
        'temperature': 0.6, 'max_length': 64
    })

    # Other tools: https://python.langchain.com/docs/integrations/tools/
    # Must pip install wikipedia, llm-math built in
    tools = load_tools(['wikipedia', 'llm-math'], llm=llm)

    # ReACT is a prompt template with "Thought, Observation, Action" stages
    # More details on ReACT (https://arxiv.org/abs/2210.03629)
    # Other agents (https://python.langchain.com/docs/modules/agents/agent_types/)
    agent = initialize_agent(tools=tools, llm=llm, verbose=True,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    return agent

my_agent = create_agent()
my_agent.run('What is the GDP per capita in Montenegro? Give the answer in " + \
              "CAD, assuming $1.30 CAD = $1 USD.')

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)
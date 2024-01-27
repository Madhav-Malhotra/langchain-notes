import os
from dotenv import load_dotenv
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.chains import APIChain

# Load environmental variable
load_dotenv(".env")

# Setup LLM
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={
    'temperature': 0.01, 'max_length': 64
})

# Setup API docs and header
docs = """
Docs for your API here.
"""

headers = {"Authorization": f"Bearer {os.environ['YOUR_API_KEY']}"}

# Create LLM chain
chain = APIChain.from_llm_and_api_docs(
    llm,
    docs,
    headers=headers,
    verbose=True,
    limit_to_domains=["https://permitted-websites.com/"],
)
chain.run(
    "What is the weather like right now in Munich, Germany?"
)
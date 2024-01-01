# import libraries
import os
from dotenv import load_dotenv

from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Need the environment variable to end with TOKEN, not KEY
load_dotenv(".env")
KEY = os.getenv('HUGGINGFACEHUB_API_KEY')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = KEY


# create a prompt template
template = """Question: {question}

Answer: Let's think step by step. 
"""
prompt = PromptTemplate(input_variables=['question'], template=template)


# load a model
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={
    'temperature': 0.6, 'max_length': 64
})


# create and run a chain
chain = LLMChain(prompt=prompt, llm=llm)
question = "Mary has 3 apples. She gives two to John and eats one." + \
" How many apples does she have left?"
out = chain.run({'question': question})
print(out)
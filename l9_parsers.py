from dotenv import load_dotenv
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser

# load model
load_dotenv(".env")
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={
    'temperature': 0.6, 'max_length': 64
})

# Create a chain
json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)
json_parser = SimpleJsonOutputParser()
json_chain = json_prompt | llm | json_parser

# Run the chain
out = json_chain.stream({'question': 'Who invented the microscope?'})
print(list(out))
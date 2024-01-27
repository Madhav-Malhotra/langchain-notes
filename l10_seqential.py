from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.llms.huggingface_hub import HuggingFaceHub


# Load key
load_dotenv(".env")

# Load model
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={
    'temperature': 0.9, 'max_length': 64
})

# Create prompts
system = "You are a helpful AI assistant that answers cooking questions."
pr_region = PromptTemplate(input_variables=['country'], template=
            system+"\nName a popular dish from {country}.")
pr_ingredient = PromptTemplate(input_variables=['dish'], template=
                system+"\nWhich ingredients are needed to make {dish}?")
pr_recipe = PromptTemplate(input_variables=['dish', 'ingredient'], template=
            system+"\nWhat are the steps to make {dish} using {ingredient}?")

# Create chains
ch_region = LLMChain(llm=llm, prompt=pr_region, output_key='dish')
ch_ingredient = LLMChain(llm=llm, prompt=pr_ingredient, output_key='ingredient')
ch_recipe = LLMChain(llm=llm, prompt=pr_recipe, output_key='recipe')

chain = SequentialChain(chains=[ch_region, ch_ingredient, ch_recipe], 
                        output_variables=['dish', 'ingredient', 'recipe'], 
                        input_variables=['country'], verbose=True)

# Run chain
out = chain.invoke({'country': 'Italy'})
print(out)
from dotenv import load_dotenv
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.evaluation import load_evaluator

# Get LLM to generate critiques of other completions. FYI This one is too small.
load_dotenv(".env")
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={
    'temperature': 0.9, 'max_length': 256
})

# use prebuilt conciseness evaluator
evaluator = load_evaluator("criteria", criteria="conciseness", llm=llm)
eval_result = evaluator.evaluate_strings(
    prediction="What's 2+2? That's an elementary question. The answer you're looking for is that two and two is four.",
    input="What's 2+2?",
)
print(eval_result)
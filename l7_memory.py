import os
from dotenv import load_dotenv

from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Load LLM
load_dotenv()
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={
    'temperature': 0.6, 'max_length': 256
})

# Load prompt
template = """You are a nice chatbot having a conversation with a human.

Previous conversation:
{chat_history}

New human question: {question}
Response:"""
prompt = PromptTemplate.from_template(template)


# Load memory/chain. Memory key based on prompt template above.
memory = ConversationBufferMemory(memory_key="chat_history")
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

# Run chain
outputs = [
    chain.run({'question': 'How are you today?'}),
    chain.run({'question': 'What is your favorite color?'}),
    chain.run({'question': 'What is your favorite food?'})
]

print(outputs) # ['I am fine, how about you?', 'I like blue', 'AI: I like seafood.']
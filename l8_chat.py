from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

model = ChatOpenAI(temperature=1, openai_api_key="YOUR_API_KEY")
chat_history = [
    SystemMessage("You are an unhelpful AI chatbot that responds sarcastically to questions."),
    HumanMessage("How can I get to New York?")
]

ai_message = model(chat_history)
print(ai_message)
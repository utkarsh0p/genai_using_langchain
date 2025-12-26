from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
   repo_id = "meta-llama/Llama-3.1-8B-Instruct",
   task="text-generation"
)

model = ChatHuggingFace(llm=llm)

chat_template = ChatPromptTemplate.from_messages([
    ('system','you are a helpful {role}'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []

with open('chat_history.txt') as f:
    store = f.readlines()
    chat_history.extend(store)


prompt = chat_template.invoke({
    'chat_history':chat_history,
    'query':'where is my refund',
    'role':'customer support agent'
})


print(model.invoke(prompt).content)

"""
    Message placeholder is special placeholder used in chatprompttemplate to insert previous chat at runtime
"""
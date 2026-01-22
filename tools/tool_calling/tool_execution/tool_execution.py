from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from langchain_community.tools import tool
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="zai-org/GLM-4.7-Flash", task="text-generation")

model = ChatHuggingFace(llm=llm)


@tool
def multiply(a, b):
    """this tool returns the multiplication of two numbers"""
    a = int(a)
    b = int(b)
    return a * b


@tool
def add(a, b):
    """this tool adds two numbers"""
    return a + b


# not every llm has capability to get binded with the tool

# messages list to maintain the chat
query = HumanMessage(content="multipy 3 and 4")
messages = [query]

model_with_tool = model.bind_tools([multiply, add])
model_with_tool_response = model_with_tool.invoke(messages)


# this is the tool message by message.invoke(<tool_call dict>)
tool_call = model_with_tool_response.tool_calls[0]
tool_message = multiply.invoke(tool_call)

# appending the ai message
messages.append(model_with_tool_response)

# appending the tool message
messages.append(tool_message)

# now giving the whole list of objects to the llm
result = model_with_tool.invoke(messages)
print(result.content)

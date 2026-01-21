from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.tools import tool
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="zai-org/GLM-4.7-Flash", task="text-generation")

model = ChatHuggingFace(llm=llm)


@tool
def multiply(a, b):
    """this tool returns the multiplication of two numbers"""
    return a * b


model_with_tool = model.bind_tools([multiply])

# not every llm has capability to get binded with the tool

model_with_tool_result = model_with_tool.invoke("multiply 3 and 5")

print(model_with_tool_result)


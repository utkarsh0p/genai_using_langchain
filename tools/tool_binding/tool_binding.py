from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.tools import tool 
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
   repo_id = "meta-llama/Llama-3.1-8B-Instruct",
   task="text-generation"
)

model = ChatHuggingFace(llm=llm)

@tool
def multiply(a,b):
    """this tool returns the multiplication of two numbers"""
    return a*b

model_with_tool = model.bind_tools([multiply])

#not every llm has capability to get binded with the tool


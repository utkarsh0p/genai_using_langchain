from langchain_community.tools import tool

@tool
def add(a , b):
    """adds two numbers"""
    return a+b

@tool
def multiply(a, b):
    """multiply two numbers"""
    return a*b

class MathToolkit:
    def get_tools(self):
        return [add, multiply]


tools = MathToolkit().get_tools()

for t in tools:
    print(t)

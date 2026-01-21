from langchain_community.tools import StructuredTool
from pydantic import BaseModel, Field

class Multiply(BaseModel):
    a:int = Field(required = True, description = "this is integer a")
    b:int = Field(required = True, description = "this is integer b")

def multiply_func(a, b):
    return a*b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply", 
    args_schema=Multiply,
    description="multiply two integers"
)


result= multiply_tool.invoke({"a":4,"b":6})
print(result)

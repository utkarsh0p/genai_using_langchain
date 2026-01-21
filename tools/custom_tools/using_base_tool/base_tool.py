from langchain_community.tools import BaseTool
from pydantic import BaseModel


class MultiplyInputSchema(BaseModel):
    a: int
    b: int


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "multiply two integers"
    args_schema : type[BaseModel] = MultiplyInputSchema

    def _run(self, a, b):
        return a * b


multiply = MultiplyTool()

result = multiply.invoke({"a": 4, "b": 5})
print(result)

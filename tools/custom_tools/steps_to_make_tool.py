from langchain_community.tools import tool
# define a function
# give types
# import tool function and add the decorator


@tool
def multiply(a: int, b: int) -> int:
    """multiply two numbers"""
    return a * b


result = multiply.invoke({"a": 4, "b": 4})

print(multiply.name)
print(multiply.args)
print(multiply.description)
#this json of the funciton goes to the model ( llm )
#-> so it can understand what the function ( tool ) does what are the inputs and wether to use or not 
print(multiply.args_schema.model_json_schema())

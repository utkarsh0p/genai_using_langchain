from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

class schema_defined(BaseModel):
    name:str = Field(description = 'name of the person ')
    age:int = Field(gt=18,description = 'age of the person')

parser = PydanticOutputParser(pydantic_object=schema_defined)

prompt = PromptTemplate(
    template="""
    Tell me about :{name}?
    {format_instructions}
    """,
    input_variables=["name"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = prompt | model | parser

result = chain.invoke({'name':'modi'})
print(result)
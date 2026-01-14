from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Cristiano_Ronaldo")
doc = loader.load()

info = doc[0].page_content #this is too long and the huggingface rejecting it so i will send small response
info = info[:3000]   # keep only first 3000 characters

prompt = PromptTemplate(
    template="My question is -> {question}\nFrom the document -> {document}",
    input_variables=["question", "document"]
)

parser = StrOutputParser()
chain = prompt | model | parser

question = "tell me how many Ballon d'Or ronaldo won?"
result = chain.invoke({"question": question, "document": info})

print(result)

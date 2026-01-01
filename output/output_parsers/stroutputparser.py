from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# stringify any type of the output model gives and give it back 
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
   repo_id = "meta-llama/Llama-3.1-8B-Instruct",
   task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template = "tell me about {name}",
    input_variables= ['name']
)


# prompt = template.invoke({
#     'name':'narendra modi'
# })

# result = model.invoke(prompt)

# parser = StrOutputParser()

# final = parser.parse(result.content)

chain = template | model | StrOutputParser()

result = chain.invoke({'name':"narendra modi"})

print(result)

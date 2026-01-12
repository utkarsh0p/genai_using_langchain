from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
   repo_id = "meta-llama/Llama-3.1-8B-Instruct",
   task="text-generation"
)

template = PromptTemplate(
    template = "give 5 interesting facts about {topic}",
    input_variables=['topic']
)

model = ChatHuggingFace(llm = llm )


# prompt = template.invoke({
#     'topic':'china'
# })

# result = model.invoke(prompt)

# print(result.content)

chain = template | model | StrOutputParser()
result = chain.invoke({'topic':"china"})

print(result)
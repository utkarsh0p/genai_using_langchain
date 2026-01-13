# this is used when we want to execute parallelly like giving a topic to models
# model1 -> for generating linkedin post
# model2 -> for generating twitter post 

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableSequence , RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
   repo_id = "meta-llama/Llama-3.1-8B-Instruct",
   task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = "create a linkedin post on topic -> {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template = "create a twitter post on topic -> {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({ # here i have use the same model for both the parallel work but we can use differen one ex- model1 and model2
    'linkedin':RunnableSequence(prompt1 | model | parser),
    'twitter':RunnableSequence(prompt2 | model | parser)
})



result = parallel_chain.invoke({'topic':'ai will help in future'})

print(result)



from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
   repo_id = "meta-llama/Llama-3.1-8B-Instruct",
   task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = "tell me about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt , model, parser)

result = chain.invoke({'topic':'random forest algorithm'})

print(result)



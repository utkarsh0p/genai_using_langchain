# runnable passthrough gives the same output as the input ( input output same , no changes)

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableSequence , RunnableParallel, RunnablePassthrough
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
    template = "create a joke on the topic -> {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template = "explain the joke -> {text}",
    input_variables=['text']
)

parser = StrOutputParser()

joke_generator = RunnableSequence(prompt1 , model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(), # same input same output 
    'joke_explanation':RunnableSequence(prompt2 , model, parser)
})


chain = RunnableSequence(joke_generator, parallel_chain)

result = chain.invoke({'topic':'cricket'})

print(result['joke'])
print(result['joke_explanation'])
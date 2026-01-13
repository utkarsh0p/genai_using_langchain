from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
   repo_id = "meta-llama/Llama-3.1-8B-Instruct",
   task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)


parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))
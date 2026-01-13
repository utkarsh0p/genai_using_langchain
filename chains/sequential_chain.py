# sequetial is also the very same as the simple but few more steps may be like calling the llm two times

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation"
)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="interesting facts about {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Give 2 most relatable and interesting fact about the {text}",
    input_variables=["text"],
)

model = ChatHuggingFace(llm=llm)


chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'china'})
print(result)

# chain.get_graph().print_ascii()  <====prints the whole flow in the terminal
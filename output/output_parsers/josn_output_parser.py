from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

# 1. Model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 2. Parser
parser = JsonOutputParser()

# 3. Prompt
prompt = PromptTemplate(
    template="""
    Give the 5 facts about :{name}?
    {format_instructions}
    """,
    input_variables=["name"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


# 4. Invoke
prompt_value = prompt.invoke({"name": "monkey"})
result = model.invoke(prompt_value)

# 5. Parse
final = parser.parse(result.content)

print(final)

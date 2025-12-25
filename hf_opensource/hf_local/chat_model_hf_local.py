from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id = "Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("who is prime minister of china")
print(result.content)

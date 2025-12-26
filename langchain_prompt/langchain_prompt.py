from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
   repo_id = "meta-llama/Llama-3.1-8B-Instruct",
   task="text-generation"
)

model = ChatHuggingFace(llm=llm)

query = st.text_input("Enter your query")

template = load_prompt('template.json')

prompt = template.invoke({
   'query':query
})


if st.button("summarize"):
   st.write(model.invoke(prompt).content)


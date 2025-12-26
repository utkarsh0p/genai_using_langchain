from langchain_core.prompts import PromptTemplate
template = PromptTemplate(
   input_variables=['query'],
   template="""Explain {query} in one line"""
)

template.save('template.json')
from langchain_community.retrievers import WikipediaRetriever
from dotenv import load_dotenv

load_dotenv()

retriever = WikipediaRetriever(top_k_results=2, lang="en")

response = retriever.invoke("Explain the theory of relativity")


print(response[0].page_content)

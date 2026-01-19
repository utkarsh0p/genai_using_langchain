from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.documents import Document
# commentng instead of deleting for now
# from langchain_community.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct"
)

embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression."),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity."),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation."),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity."),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy."),
]

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./database/mqr_db",
    collection_name="mqr_collection",
)

vector_store.add_documents(docs)

# mqr_retriever = MultiQueryRetriever.from_llm(
#     retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
#     llm=llm
# )

# result = mqr_retriever.invoke("How can I improve my mental and physical health?")

# print(result)


# currently my langchian version may be not supporting this mqr retriever properly
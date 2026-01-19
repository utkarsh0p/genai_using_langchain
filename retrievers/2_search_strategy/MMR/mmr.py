from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)
# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(
        page_content="MMR helps you get diverse results when doing similarity search."
    ),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
    Document(
        page_content="HuggingFace provides many pre-trained models for embeddings."
    ),
]


vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./database/mmr_db",
    collection_name="mmr_collection",
)
vector_store.add_documents(docs)
retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.5}
)

result = retriever.invoke("What is LangChain?")
print(result)
print(type(result))

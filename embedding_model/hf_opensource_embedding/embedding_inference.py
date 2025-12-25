from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

# vector = embeddings.embed_query("Where is Delhi?")

documents = [
    "How to learn things faster",
    "I like to learn new tech "
]

vector = embeddings.embed_documents(documents)
print(str(vector))

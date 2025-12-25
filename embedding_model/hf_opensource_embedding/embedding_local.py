from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

query = "Where is Delhi"

vector = embedding.embed_query(query)
print(str(vector))
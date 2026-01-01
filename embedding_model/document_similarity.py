from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

document = [
    "virat is highest run scorer",
    "Dhoni is captain of csk",
    "Modi is prime minister of India",
    "India is the most populated country of the world",
    "satyadarsh lund"
]


query = "satyadarsh kon hai"

document_vector = embedding.embed_documents(document)
query_vector = embedding.embed_query(query)

scores = cosine_similarity([query_vector], document_vector)[0]

lscores = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

index, score = lscores

print(document[index])
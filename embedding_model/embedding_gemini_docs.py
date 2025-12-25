from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)

documents = [
    "Who is Superman",
    "Was Thanos right to wipe out half of population"
]

vector = embeddings.embed_documents(documents)
print(str(vector))
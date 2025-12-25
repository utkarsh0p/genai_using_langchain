from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)
#  we can also send documents and use embed.documents
vector = embeddings.embed_query("Who is the prime minister of India?")

print(str(vector))
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm = GoogleGenerativeAI(model="models/gemini-3-flash-preview")
response = llm.invoke("who are you")
print(response)

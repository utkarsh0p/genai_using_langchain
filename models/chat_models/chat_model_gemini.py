from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="models/gemini-3-flash-preview")
response = model.invoke("who is modi")

print(response.content[0]['text'])


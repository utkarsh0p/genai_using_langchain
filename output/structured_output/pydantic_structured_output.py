from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="models/gemini-3-flash-preview")

# defining the BaseModel schema
class Review(BaseModel):
    animal_name:str 
    about:list[str]= Field(default='none', description="write all about inside the list of two items")
    behaviour: str


# Enable structured output
# giving the schema of the output to the model
structured_model = model.with_structured_output(Review)

# Invoke
result = structured_model.invoke("What is a tiger?")

# Result is already structured (dict)
print(result)





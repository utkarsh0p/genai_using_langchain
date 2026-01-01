from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="models/gemini-3-flash-preview")

# defining the typedict schema
class Review(TypedDict):
    animal_name:Annotated[str, 'name of the animal']
    # annoted is use to explicitly define what the key is meant to output
    behaviour: str


# Enable structured output
# giving the schema of the output to the model
structured_model = model.with_structured_output(Review)

# Invoke
result = structured_model.invoke("What is a tiger?")

# Result is already structured (dict)
print(result['about'])

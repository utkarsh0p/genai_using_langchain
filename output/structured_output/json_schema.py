# this is json_schema this is used because if the project have more than one language then
#  we can not use the pydantic or the typedict

from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="models/gemini-3-flash-preview")


# defining the  schema
json_schema = {
    "name":{
        "type":"string",
        "description":"give the name of the animal in the topic"
    },
    "pros":{
        "type":["array"],
        "items":{
            "type":"string"
        },
        "description":"give the pros of the anima"
    }

}


# Enable structured output
# giving the schema of the output to the model
structured_model = model.with_structured_output(json_schema)

# Invoke
result = structured_model.invoke("What is a tiger?")

# Result is already structured (dict)
print(result['description'])





from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

parser = StrOutputParser()   # parser (string)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation"
)

modelOne = ChatGoogleGenerativeAI(model="models/gemini-3-flash-preview")
modelTwo = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template = "Generate short and simple note from the following text {text}",
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template="Generate 5 short question and answer fromt the following text {text}",
    input_variables =['text']
)

prompt3 = PromptTemplate(
    template = "Merge the provided notes->{notes} and quiz->{quiz} into the sigle document",
    input_variables=['notes', 'quiz']
)

parallel_chain = RunnableParallel({
    'notes': prompt1 | modelOne | parser,
    'quiz': prompt2 | modelTwo | parser
})

merge_chain = prompt3 | modelOne | parser

chain =parallel_chain | merge_chain

text = """
 India's history spans over 5,000 years, beginning with the Indus Valley Civilization, one of the world's earliest urban cultures, which flourished from approximately 3300 BCE to 1300 BCE.
 This was followed by the Vedic period, marked by the arrival of the Aryans and the composition of the Vedas, laying the foundation for Hinduism.
 The Maurya Empire, particularly under Emperor Ashoka in the 3rd century BCE, unified much of the Indian subcontinent and promoted Buddhism across Asia.
 The Gupta Empire in the 4th to 6th centuries CE is often referred to as the Golden Age of Indian history, witnessing significant advancements in science, mathematics, and the arts.
 The medieval era saw the rise of powerful Islamic dynasties, including the Delhi Sultanate and the Mughal Empire, which ruled India for centuries and left a lasting legacy in architecture and governance.
 European colonial influence began in the 16th century, culminating in British rule under the British Raj, which lasted until India gained independence in 1947 after a long struggle led by figures like Mahatma Gandhi and Jawaharlal Nehru.
"""

result = chain.invoke({'text':text})

print(result)

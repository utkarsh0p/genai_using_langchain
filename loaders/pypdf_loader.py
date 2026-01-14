# the each page of pdf will be coverted to the document ( ex- 4 page pdf list of 4 document each document of one page)
# it uses pypdf library ( this is used mainly for the simple documents)

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
   repo_id = "meta-llama/Llama-3.1-8B-Instruct",
   task="text-generation"
)

model = ChatHuggingFace(llm=llm)

loader = PyPDFLoader("loaders/Learn_PyPDF_Basics.pdf")
docs = loader.load()


print(docs[0].page_content)

#limitation
# it will not work properly in many cases like scanned images

# so suppose we have to load a whole foler which have pdfs ( scenario something like this ) in that case we use directory loader

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation"
)

model = ChatHuggingFace(llm=llm)

loader = DirectoryLoader(
    path="loaders",  # this is the path
    glob="*.pdf",  # your preference whatever you want so select here all the pdfs will be selected
    loader_cls=PyPDFLoader,  # this is the loader class for all the pdf  files
)

docs = loader.load()

print(docs[0].page_content) #<- pdf 1 which has 5 pages this is the first page ( document )
print(docs[5].page_content) #<- pdf 2 which has 5 pages this is the first page ( document ) 


# the problem is that it takes too much time to load( not here but suppose i have big collection )
# that is why we use lazy loading
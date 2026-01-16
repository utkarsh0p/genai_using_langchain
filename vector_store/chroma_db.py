from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
#there are many vectore store like faiss, pinecone etc
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)


# Create LangChain documents for IPL players

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )


vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='my_chroma_db',
    collection_name='sample'
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store.add_documents(docs)

show = vector_store.get(include=['embeddings','documents', 'metadatas'])
#show is the directory here and it will have the 
# {
#   "ids": [...],
#   "documents": [...],
#   "metadatas": [...],
#   "embeddings": [...]
# }

query1 = vector_store.similarity_search(
    query = "who is the best batter",
    k=1
)


#CRUD OPERARION
# ex = to update the db we will use update_document and use the same id of that document to update

print(query1)



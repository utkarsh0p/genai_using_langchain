from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size = 10,
    #between two chunks if you want to keep some info similar use chunk overlap
    chunk_overlap=0,
    separator=""
)

sentence ="""
    my name is utkarsh singh
    and i want to be a software engineer
    when i was in class 12 i came to know about this cs field and soon enough i took interest in it
"""

result = splitter.split_text(sentence) 
print(result)
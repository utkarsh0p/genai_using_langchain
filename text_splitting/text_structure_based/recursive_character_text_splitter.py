from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap =0,
)

sentence = """
    Earth is the third planet from the Sun and the only known place where life exists. 
    It has land, water, and air that support plants, animals, and humans. 
    About 70% of Earth’s surface is covered by oceans, which help control the planet’s temperature. 
    The Earth also has a protective atmosphere that shields us from harmful space radiation. 
    Because of these features, Earth is a unique and special planet in our solar system.
"""

result  = splitter.split_text(sentence)
print(result)


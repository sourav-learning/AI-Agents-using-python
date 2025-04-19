import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

#fetch the API key from the environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Initialize the OpenAIEmbeddings with the model and temperature
llm = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# Load the text file containing job listings
document = TextLoader("./input/job_listings.txt").load()
# Split the document into smaller chunks for processing
# The chunk size is set to 200 characters with an overlap of 10 characters between chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
# Split the document into chunks
doc_chunk = text_splitter.split_documents(document)

# Create a vector store from the document chunks using the OpenAI embeddings    
# The Chroma vector store is used to store the embeddings for efficient retrieval. It is in memory store
db = Chroma.from_documents(doc_chunk, llm)
# Optional - Save the vector store to disk for later use. Persistant
#db = Chroma.from_documents(doc_chunk, llm, persist_directory="./chroma_db")
#db.persist()
# retriever is created which will search and retrieve the relevant chunks from the vector store based on the query
retriever = db.as_retriever()
# The search is performed using the retriever and the query is passed to it
search_query = input("Enter your job search query: ")
# Perform the search using the retriever
results = retriever.invoke(search_query)
# Print the results of the search
for result in results:
    print(result.page_content)

                           
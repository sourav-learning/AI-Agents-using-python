# This python script will take input query and search a pdf file containing leave related details to 
# answer the user query. 
# This  interaction should be history enabled.

import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# open api key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY ")
# embeddings 
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# load the pdf file and split it into chunks
document = PyPDFLoader("./input/Employee_Leave_policy.pdf").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks = text_splitter.split_documents(document)

# create a vector store from the chunks using embeddings
db = Chroma.from_documents(chunks, embeddings)
# chunks to be retrieved from the database
retriever = db.as_retriever()

# create a retriever with a search query and a vector store
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","""You are an HR assistant of ABC Corp. Please respond to users query related to leave policy 
         based on the context provided. If the answer is not clear, then tell user that you do not have the required info and user 
         need to email hr@abc.com. Your response should be concise within 2 to 3 sentences and specific to the question asked.
         {context}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_template)
qa_chain = create_stuff_documents_chain(llm, prompt_template)

# create a retrieval chain that uses the retriever and the qa_chain
# The retrieval chain will first retrieve relevant documents from the vector store and then use the qa_chain to generate a response
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

history_for_chain= StreamlitChatMessageHistory()

# Streamlit app to take user input and display the response from the assistant
st.title("Leave Policy Assistant")
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)
st.write("Ask your query related to leave policy")
question = st.text_input("Enter your question:")
if question:
    response = chain_with_history.invoke({"input": question},
                                         {"configurable": {"session_id": "abcd"}})
    st.write(response["answer"])
    

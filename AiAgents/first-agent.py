import os
import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.9, openai_api_key=OPENAI_API_KEY)
prompt="You are a helpful assistant. Answer the following question: {input}"
#input = input("Enter your question: ")  # To prrompt user for providing input when not using streamlit
st.title("My First AI assistant")
input = st.text_input("Enter your question: ")
if input:
    response = llm.invoke(prompt.format(input=input))
    st.write(response.content)


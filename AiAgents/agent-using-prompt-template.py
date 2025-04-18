import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, openai_api_key=OPENAI_API_KEY)
prompt_template  = PromptTemplate(
    input_variables=["input"],
    template="You are a senior python developer. Write a python code as per the following requirement : {input}?",
)

st.title("Python Code Generator")
st.write("Enter your requirement below:")
user_input = st.text_input("Requirement ")
if user_input:
    response = llm.invoke(prompt_template.format(input=user_input))
    st.write("Generated Python Code:")
    st.code(response, language='python')
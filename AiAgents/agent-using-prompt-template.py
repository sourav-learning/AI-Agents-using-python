import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# fetch the API key from the environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize the ChatOpenAI with the model and temperature
# The model is set to "gpt-4o-mini" and the temperature is set to 0.9 for more creative responses
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, openai_api_key=OPENAI_API_KEY)
# Create a prompt template for generating Python code
prompt_template  = PromptTemplate(
    input_variables=["input"],
    template="You are a senior python developer. Write a python code as per the following requirement : {input}?",
)
# Create a Streamlit app for user input and displaying the generated code
st.title("Python Code Generator")
st.write("Enter your requirement below:")
user_input = st.text_input("Requirement ")
# When the user submits the input, generate the Python code using the LLM
if user_input:
    response = llm.invoke(prompt_template.format(input=user_input))
    st.write("Generated Python Code:")
    st.code(response, language='python')
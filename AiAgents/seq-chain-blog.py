import os
import logging
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Configure logging to print to the console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_API_KEY)

title_prompt_template  = PromptTemplate(
    input_variables=["topic","audience"],
    template="You are a senior content writer. Write a catchy title for a blog post on the topic '{topic}' that is targeted towards '{audience}'.",
)

title_chain = title_prompt_template | llm | StrOutputParser() | (lambda title  : (st.write(title), title)[1])

content_prompt_template  = PromptTemplate(
    input_variables=["title","audience","noOfSentences"],
    template="You are a senior content writer. Write a detailed blog post based on the title '{title}' " \
    "that is targeted towards '{audience}' and contains {noOfSentences} sentences. The write up should have a professional tone "
    "and be informative and engaging.")
    
content_chain = content_prompt_template | llm 
final_chain = title_chain | (lambda title: {"title": title, "audience": audience, "noOfSentences": noOfSentences}) |content_chain   

topic = st.text_input("Topic")
audience = st.text_input("Audience")
noOfSentences = st.number_input("Number of Sentences", min_value=1, max_value=100, value=5)
# Print debugging info to the console
logging.debug(f"Topic: {topic}, Audience: {audience}, No. of Sentences: {noOfSentences}")
if topic and audience:
    try:
        # Print debugging info to the console
        logging.debug(f"Topic: {topic}, Audience: {audience}, No. of Sentences: {noOfSentences}")
        response = final_chain.invoke({"topic": topic, "audience": audience, "noOfSentences": noOfSentences})
        logging.info("Generated Blog Post:")
        st.write(response.content)
    except Exception as e:
        # Print the error to the console
        logging.error(f"An error occurred: {e}", exc_info=True)
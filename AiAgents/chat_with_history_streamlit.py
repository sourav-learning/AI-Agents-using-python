import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

prompt_template = ChatPromptTemplate.from_messages(
[
    ("system", "Consider the role of a career consultant for fresher or experienced job seekers. "
                "Based on the question, answer to the best of your knowledge."
                "If you do not know  how to answer, tell that. Make sure to suggest something that is relevant to the job search "
                "and logic behind it. You can suggest relevant courses, certifications, or skills to learn. "),
    MessagesPlaceholder(variable_name = "chat_history"),
    ("human", "{question}")
]
)
# The prompt template is used to create a chain that will be used to generate responses
chain = prompt_template | llm
# The chat message history is used to keep track of the conversation history
history_for_chain = StreamlitChatMessageHistory()
final_chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,
    input_messages_key = "question",
    history_messages_key = "chat_history"
)

st.title("Welcome to the Career assistant. Type 'exit' to quit.")

query = st.text_input("You: ")
if query :
    # Run the chain with the user's question and the chat history
    response = final_chain_with_history.invoke({"question": query},
                                            {"configurable": {"session_id": "test123"}})
    # Print the response from the assistant
    st.write("AI Career Advisor:", response.content)



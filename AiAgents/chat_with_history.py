import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

prompt_template = ChatPromptTemplate.from_messages(
[
    ("system", "Consider the role of a financial assistant. Based on the question, answer to the best of your knowledge."
                "If you do not know  how to answer, tell that. If you are suggesting something, provide the calculation "
                "and logic behind it. Make sure not to recommend any company product or stock which you are not allowed to promote"),
    MessagesPlaceholder(variable_name = "chat_history"),
    ("human", "{question}")
]
)
# The prompt template is used to create a chain that will be used to generate responses
chain = prompt_template | llm
# The chat message history is used to keep track of the conversation history
history_for_chain = ChatMessageHistory()
final_chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,
    input_messages_key = "question",
    history_messages_key = "chat_history"
)

print("Welcome to the financial assistant. Type 'exit' to quit.")
while True:
    query = input("You: ")
    if query == "exit":
        break
    # Run the chain with the user's question and the chat history
    response = final_chain_with_history.invoke({"question": query},
                                               {"configurable": {"session_id": "test123"}})
    # Print the response from the assistant
    print("AI Financial Advisor:", response.content)



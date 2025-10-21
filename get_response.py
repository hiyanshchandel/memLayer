from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    temperature=0.7,
    model=os.getenv("OPENAI_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

# Single conversation history (no multiple sessions)
conversation_history = ChatMessageHistory()

def get_session_history(session_id: str):
    return conversation_history

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

def get_chatbot_response(user_input: str) -> str:
    response = conversation.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "default"}}
    )
    return response.content

def get_history():
    history = []
    for message in conversation_history.messages[:10]:
        role = "User" if message.type == "human" else "AI"
        history.append((role, message.content))
    return history


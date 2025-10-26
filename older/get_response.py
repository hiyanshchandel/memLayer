from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from .deduplication import get_similar
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    temperature=0.7,
    model=os.getenv("OPENAI_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to past conversation memories.
{context}
Use the relevant memories above (if any) to provide more personalized and contextual responses.
If the memories are relevant to the current conversation, incorporate them naturally.
If they're not relevant, you can ignore them and respond normally."""),
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
    similar_memories, _, _ = get_similar(user_input, threshold=0.25)
    context = "\n".join(similar_memories) if similar_memories else "No relevant past memories found."
    response = conversation.invoke(
        {"input": user_input, "context": context},
        config={"configurable": {"session_id": "default"}}
    )
    return response.content

def get_history():
    history = []
    for message in conversation_history.messages[-10:] :  # Get LAST 10 messages
        role = "User" if message.type == "human" else "AI"
        history.append((role, message.content))
    return history


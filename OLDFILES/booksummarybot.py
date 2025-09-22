import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# API key setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not set. Please add it to your .env file.")
    st.stop()


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! Type the name of any book and I'll summarize it for you."}
    ]


memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3, return_messages=True)

prompt_template = """
You are a helpful book summary bot.

Your job:
- When a user types the name of a book, provide a concise summary of that book.
- If the user asks for another book, summarize that one.
- If you don't know the book, politely say so.
- Use conversation history to avoid repeating summaries.

Conversation history:
{chat_history}

User’s latest request: {input}
Bot:
"""

# LLM, Prompt and conversation chain
prompt = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=prompt_template
)


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)


# title and message 
st.title("📚 Book Summary Bot")


for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
user_input = st.chat_input("Ask me about a book...")
if user_input:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.spinner("Summarizing..."):
        try:
            response = conversation.predict(input=user_input)
        except Exception as e:
            response = f"⚠️ Error: {str(e)}"



    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)


st.sidebar.caption("⚙️ Config")
st.sidebar.write("API key loaded:", bool(GOOGLE_API_KEY))




#chat_utils.py
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.base import Runnable
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

class ChatAgent:
    def __init__(self, prompt: ChatPromptTemplate, llm: Runnable):
        """
        Initialize the ChatAgent.
        """
        self.history = StreamlitChatMessageHistory(key="chat_history")
        self.llm = llm
        self.prompt = prompt
        self.chain = self.setup_chain()

    def setup_chain(self) -> RunnableWithMessageHistory:
        """
        Set up the chain for the ChatAgent.
        """
        chain = self.prompt | self.llm
        return RunnableWithMessageHistory(
            chain,
            lambda session_id: self.history,
            input_messages_key="question",  # This must match your prompt input
            history_messages_key="history"  # Must match MessagesPlaceholder name
        )

    def display_messages(self):
        """
        Display chat messages in the Streamlit UI.
        """
        if len(self.history.messages) == 0:
            self.history.add_ai_message("How can I help you?")
        for msg in self.history.messages:
            st.chat_message(msg.type).write(msg.content)

    def start_conversation(self):
        """
        Run the Streamlit chatbot.
        """
        self.display_messages()
        user_question = st.chat_input(placeholder="Ask me anything!")

        if user_question:
            st.chat_message("human").write(user_question)
            config = {"configurable": {"session_id": "session"}}

            try:
                response = self.chain.invoke({"question": user_question}, config)
                if hasattr(response, "content"):
                    st.chat_message("ai").write(response.content)
                else:
                    st.chat_message("ai").write(str(response))
            except Exception as e:
                st.error(f"An error occurred: {e}")
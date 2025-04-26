# app/tests/tests_chat_utils.py

import unittest
from unittest.mock import MagicMock, patch
from app.components.chat_utils import ChatAgent
from app.components.prompts import chat_prompt_template

class TestChatAgent(unittest.TestCase):
    def setUp(self):
        # Mock the LLM runnable
        self.mock_llm = MagicMock()
        self.mock_llm.invoke.return_value = MagicMock(content="Mock response")

        # Instantiate the agent
        self.agent = ChatAgent(prompt=chat_prompt_template, llm=self.mock_llm)

    @patch("app.components.chat_utils.st")
    def test_display_messages_empty_history(self, mock_st):
        # Ensure the chat history is initially empty
        self.agent.history.clear()
        self.agent.display_messages()

        # Check that the AI default message was added
        self.assertEqual(self.agent.history.messages[0].content, "How can I help you?")
        mock_st.chat_message.assert_called()  # Check if streamlit was called to write messages

    @patch("app.components.chat_utils.st")
    def test_start_conversation_flow(self, mock_st):
        # Simulate a user input
        mock_st.chat_input.return_value = "What is LangChain?"
        mock_st.chat_message.return_value.write = MagicMock()

        # Run the conversation method
        self.agent.start_conversation()

        # Check that the mock LLM was called with the expected question
        self.mock_llm.invoke.assert_called()
        args, kwargs = self.mock_llm.invoke.call_args
        self.assertIn("question", args[0])
        self.assertEqual(args[0]["question"], "What is LangChain?")

if __name__ == "__main__":
    unittest.main()

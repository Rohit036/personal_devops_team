from pydantic import BaseModel
from agents.base_agent import BaseDevOpsAgent
from utils.groq_client import GROQClient
from models.groq_models import ChatCreateRequest, ChatCreateResponse
from github import Github
import os
from typing import Dict, Any

class ChatAgentConfig(BaseModel):
    """
    Configuration settings for the Chat agent.
    
    Attributes:
        chat_model_id (str): Identifier for the chat model to be used
        groq_api_endpoint (str): GROQ API endpoint URL
        groq_api_key (str): Authentication key for GROQ API
        github_token (str): GitHub authentication token
        repo_name (str): GitHub repository name in format "username/repo"
        pull_request_number (int): PR number to analyze and comment on
    """
    chat_model_id: str
    groq_api_endpoint: str
    groq_api_key: str
    github_token: str
    repo_name: str  # e.g., "username/repo"
    pull_request_number: int

class ChatAgent(BaseDevOpsAgent):
    """
    An AI agent that interacts with GitHub pull requests using GROQ's language models.
    
    This agent can analyze pull requests, provide feedback, and post comments directly
    to GitHub using AI-generated responses.
    """

    def __init__(self, config: ChatAgentConfig):
        """
        Initialize the Chat agent with necessary clients and configuration.
        
        Args:
            config (ChatAgentConfig): Configuration object containing API keys and settings
        """
        self.config = config
        self.groq_client = GROQClient(
            api_endpoint=config.groq_api_endpoint,
            api_key=config.groq_api_key
        )
        self.github_client = Github(config.github_token)

    def fetch_pull_request_files(self):
        """
        Retrieve the files modified in the specified pull request.
        
        Returns:
            PaginatedList: List of files modified in the pull request
        """
        repo = self.github_client.get_repo(self.config.repo_name)
        pull_request = repo.get_pull(self.config.pull_request_number)
        files = pull_request.get_files()
        return files

    def perform_chat_interaction(self, user_message: str, context: Dict[str, Any] = None) -> ChatCreateResponse:
        """
        Send a message to the GROQ API and get an AI-generated response.
        
        Args:
            user_message (str): The message to send to the AI
            context (Dict[str, Any], optional): Additional context for the conversation
        
        Returns:
            ChatCreateResponse: The AI's response and metadata
        
        Raises:
            Exception: If the chat interaction fails
        """
        chat_request = ChatCreateRequest(
            user_message=user_message,
            context=context
        )
        try:
            response = self.groq_client.send_chat_create_request(chat_request)
            return response
        except Exception as e:
            print(f"Error during chat interaction: {e}")
            raise

    def post_feedback_to_github(self, bot_response: str):
        """
        Post the AI's response as a comment on the GitHub pull request.
        
        Args:
            bot_response (str): The AI-generated response to post
        """
        repo = self.github_client.get_repo(self.config.repo_name)
        pull_request = repo.get_pull(self.config.pull_request_number)
        comment = f"🤖 **AI Assistant:** {bot_response}"
        pull_request.create_issue_comment(comment)

    def run(self):
        """
        Execute the main workflow of the chat agent.
        
        This method:
        1. Sends a request to review the pull request
        2. Gets an AI-generated response
        3. Posts the feedback to GitHub
        
        Returns:
            Dict: Contains the bot's response, confidence score, and status
                 or an error message if the interaction fails
        """
        # Example: Ask the AI assistant to review the pull request
        user_message = "Please review the recent changes in this pull request for code quality and potential issues."
        response = self.perform_chat_interaction(user_message)
        
        if response.status == "success":
            bot_response = response.bot_response
            self.post_feedback_to_github(bot_response)
            return {
                "bot_response": bot_response,
                "confidence": response.confidence,
                "status": response.status
            }
        else:
            return {"error": "Failed to get a successful response from GROQ Chat-Create API."}
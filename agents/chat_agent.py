from pydantic import BaseModel
from agents.base_agent import BaseDevOpsAgent
from utils.azure_openai_client import AzureOpenAIClient
from github import Github
import os
from typing import Dict, Any

class ChatAgentConfig(BaseModel):
    """
    Configuration settings for the Chat agent.
    
    Attributes:
        azure_openai_endpoint (str): Azure OpenAI endpoint URL
        azure_openai_key (str): Azure OpenAI API key
        azure_openai_deployment (str): Azure model deployment name
        azure_openai_api_version (str): Azure OpenAI API version
        github_token (str): GitHub authentication token
        repo_name (str): GitHub repository name in format "username/repo"
        pull_request_number (int): PR number to analyze and comment on
    """
    azure_openai_endpoint: str = ""
    azure_openai_key: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-12-01-preview"
    github_token: str
    repo_name: str  # e.g., "username/repo"
    pull_request_number: int

    @classmethod
    def from_env(cls, repo_name: str, pull_request_number: int) -> "ChatAgentConfig":
        return cls(
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            repo_name=repo_name,
            pull_request_number=pull_request_number,
        )

class ChatAgent(BaseDevOpsAgent):
    """
    An AI agent that interacts with GitHub pull requests using Azure OpenAI.
    
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
        self.azure_client = AzureOpenAIClient(
            endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_key,
            deployment_name=config.azure_openai_deployment,
            api_version=config.azure_openai_api_version,
            temperature=0.4,
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

    def perform_chat_interaction(self, user_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send a message to Azure OpenAI and get an AI-generated response.
        
        Args:
            user_message (str): The message to send to the AI
            context (Dict[str, Any], optional): Additional context for the conversation
        
        Returns:
            ChatCreateResponse: The AI's response and metadata
        
        Raises:
            Exception: If the chat interaction fails
        """
        try:
            full_message = user_message
            if context:
                full_message += f"\n\nContext: {context}"
            bot_response = self.azure_client.chat(
                system_prompt="You are a helpful GitHub pull request assistant.",
                user_message=full_message,
            )
            return {
                "bot_response": bot_response,
                "confidence": 0.8,
                "status": "success",
            }
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
        
        if response.get("status") == "success":
            bot_response = response.get("bot_response", "")
            self.post_feedback_to_github(bot_response)
            return {
                "bot_response": bot_response,
                "confidence": response.get("confidence", 0.0),
                "status": response.get("status", "error"),
            }
        else:
            return {"error": "Failed to get a successful response from Azure OpenAI."}
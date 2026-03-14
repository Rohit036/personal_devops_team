from pydantic import BaseModel
from agents.base_agent import BaseDevOpsAgent
from utils.groq_client import GROQClient
from models.groq_models import CodeReviewRequest, CodeReviewFeedback
from github import Github
import os

class CodeReviewConfig(BaseModel):
    """
    Configuration settings for the Code Review agent.
    
    Attributes:
        model (str): The LLM model to use for code review (default: llama3-8b-8192)
        groq_api_endpoint (str): GROQ API endpoint URL
        groq_api_key (str): Authentication key for GROQ API
        github_token (str): GitHub authentication token
        repo_name (str): GitHub repository name in format "username/repo"
        pull_request_number (int): PR number to review
    """
    model: str = "llama3-8b-8192"  # Default model for code review
    groq_api_endpoint: str
    groq_api_key: str
    github_token: str
    repo_name: str
    pull_request_number: int

class CodeReviewAgent(BaseDevOpsAgent):
    """
    An AI agent that performs automated code reviews on GitHub pull requests.
    
    This agent analyzes Python files in pull requests, provides feedback on code quality,
    and posts detailed review comments directly to GitHub.
    """

    def __init__(self, config: CodeReviewConfig):
        """
        Initialize the Code Review agent with necessary clients and configuration.
        
        Args:
            config (CodeReviewConfig): Configuration object containing API keys and settings
        """
        self.config = config
        self.groq_client = GROQClient(
            api_endpoint=config.groq_api_endpoint,
            api_key=config.groq_api_key
        )
        self.github_client = Github(self.config.github_token)

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

    def perform_code_review(self):
        """
        Analyze modified Python files in the pull request and generate review feedback.
        
        The method:
        1. Fetches modified files from the pull request
        2. Analyzes Python files using the GROQ API
        3. Generates detailed feedback for each file
        
        Returns:
            list: List of dictionaries containing feedback for each reviewed file
                 Including issues found, suggestions, and overall quality scores
        """
        files = self.fetch_pull_request_files()
        feedback = []

        for file in files:
            if file.filename.endswith('.py'):  # Focus on Python files
                file_content = file.patch  # Get the diff
                # Create review request for the file
                code_review_request = CodeReviewRequest(
                    file_name=file.filename,
                    file_content=file.raw_url,  # You might need to fetch the actual content
                    diff=file.patch
                )
                try:
                    # Send the review request to GROQ API
                    review_feedback = self.groq_client.send_code_review_request(
                        model_id=self.config.model,
                        code_review_request=code_review_request
                    )
                    feedback.append({
                        "file": file.filename,
                        "issues": review_feedback.issues,
                        "suggestions": review_feedback.suggestions,
                        "overall_quality": review_feedback.overall_quality
                    })
                except Exception as e:
                    feedback.append({
                        "file": file.filename,
                        "error": str(e)
                    })

        return feedback

    def post_feedback_to_github(self, feedback):
        """
        Post the code review feedback as comments on the GitHub pull request.
        
        Args:
            feedback (list): List of feedback dictionaries for each reviewed file
                           containing issues, suggestions, and quality scores
        """
        repo = self.github_client.get_repo(self.config.repo_name)
        pull_request = repo.get_pull(self.config.pull_request_number)

        for file_feedback in feedback:
            if "error" in file_feedback:
                # Handle error cases with warning message
                comment = f"⚠️ **Code Review Error**: {file_feedback['error']}"
            else:
                # Format successful review feedback
                issues = "\n".join([f"- {issue['description']}" for issue in file_feedback['issues']])
                suggestions = "\n".join([f"- {suggestion}" for suggestion in file_feedback['suggestions']])
                overall = file_feedback['overall_quality']

                comment = (
                    f"### 📝 Code Review for `{file_feedback['file']}`\n\n"
                    f"**Overall Quality**: {overall}\n\n"
                    f"**Issues Found**:\n{issues}\n\n"
                    f"**Suggestions**:\n{suggestions}"
                )
            # Post the comment on the pull request
            pull_request.create_issue_comment(comment)

    def run(self):
        """
        Execute the main workflow of the code review agent.
        
        This method:
        1. Performs code review on the pull request files
        2. Posts the feedback to GitHub
        3. Returns the complete feedback data
        
        Returns:
            list: Complete feedback data for all reviewed files
        """
        feedback = self.perform_code_review()
        self.post_feedback_to_github(feedback)
        return feedback
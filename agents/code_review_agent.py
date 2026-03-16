import json
import os

from pydantic import BaseModel

from agents.base_agent import BaseDevOpsAgent
from utils.azure_openai_client import AzureOpenAIClient
from github import Github

class CodeReviewConfig(BaseModel):
    """
    Configuration settings for the Code Review agent.
    
    Attributes:
        azure_openai_endpoint (str): Azure OpenAI endpoint URL
        azure_openai_key (str): Azure OpenAI API key
        azure_openai_deployment (str): Azure model deployment name
        azure_openai_api_version (str): Azure OpenAI API version
        github_token (str): GitHub authentication token
        repo_name (str): GitHub repository name in format "username/repo"
        pull_request_number (int): PR number to review
    """
    azure_openai_endpoint: str = ""
    azure_openai_key: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-12-01-preview"
    github_token: str
    repo_name: str
    pull_request_number: int

    @classmethod
    def from_env(cls, repo_name: str, pull_request_number: int) -> "CodeReviewConfig":
        return cls(
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            repo_name=repo_name,
            pull_request_number=pull_request_number,
        )

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
        self.azure_client = AzureOpenAIClient(
            endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_key,
            deployment_name=config.azure_openai_deployment,
            api_version=config.azure_openai_api_version,
            temperature=0.2,
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
        2. Analyzes Python files using Azure OpenAI
        3. Generates detailed feedback for each file
        
        Returns:
            list: List of dictionaries containing feedback for each reviewed file
                 Including issues found, suggestions, and overall quality scores
        """
        files = self.fetch_pull_request_files()
        feedback = []

        for file in files:
            if file.filename.endswith('.py'):  # Focus on Python files
                file_content = file.patch or ""
                try:
                    review_feedback = self._review_diff_with_azure(
                        file_name=file.filename,
                        diff=file_content,
                    )
                    feedback.append({
                        "file": file.filename,
                        "issues": review_feedback.get("issues", []),
                        "suggestions": review_feedback.get("suggestions", []),
                        "overall_quality": review_feedback.get("overall_quality", "unknown"),
                    })
                except Exception as e:
                    feedback.append({
                        "file": file.filename,
                        "error": str(e)
                    })

        return feedback

    def _review_diff_with_azure(self, file_name: str, diff: str) -> dict:
        prompt = (
            "Review this Python git diff and return strict JSON with keys: "
            "issues (array of {description,severity}), suggestions (array of strings), "
            "overall_quality (one of high|medium|low).\n"
            f"File: {file_name}\nDiff:\n{diff}"
        )
        response = self.azure_client.chat(
            system_prompt="You are a senior Python code reviewer. Return JSON only.",
            user_message=prompt,
        )
        cleaned = response.strip()
        if cleaned.startswith("```") and "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Invalid review response format")
        return parsed

    def post_feedback_to_github(self, feedback):
        """
        Post the code review feedback as comments on the GitHub pull request.
        
        Args:
            feedback (list): List of feedback dictionaries for each reviewed file
                           containing issues, suggestions, and quality scores
        """
        repo = self.github_client.get_repo(self.config.repo_name)
        pull_request = repo.get_pull(self.config.pull_request_number)
        marker = "<!-- ai-code-review -->"
        sections = [marker, "##  AI Code Review Summary", ""]

        if not feedback:
            sections.append("No Python files were changed in this PR.")
        else:
            for file_feedback in feedback:
                sections.append(f"### File: {file_feedback['file']}")
                if "error" in file_feedback:
                    sections.append(f"-  Error: {file_feedback['error']}")
                    sections.append("")
                    continue

                overall = file_feedback.get("overall_quality", "unknown")
                sections.append(f"- Overall Quality: {overall}")

                issues = file_feedback.get("issues", [])
                if issues:
                    sections.append("- Issues Found:")
                    for issue in issues:
                        description = issue.get("description", "Unspecified issue")
                        severity = issue.get("severity", "unspecified")
                        sections.append(f"  - [{severity}] {description}")
                else:
                    sections.append("- Issues Found: none")

                suggestions = file_feedback.get("suggestions", [])
                if suggestions:
                    sections.append("- Suggestions:")
                    for suggestion in suggestions:
                        sections.append(f"  - {suggestion}")
                else:
                    sections.append("- Suggestions: none")

                sections.append("")

        sections.append("---")
        sections.append("_This comment is automatically updated on each run._")
        comment_body = "\n".join(sections)

        existing_comment = None
        for comment in pull_request.get_issue_comments():
            if marker in (comment.body or ""):
                existing_comment = comment
                break

        if existing_comment:
            existing_comment.edit(comment_body)
        else:
            pull_request.create_issue_comment(comment_body)

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
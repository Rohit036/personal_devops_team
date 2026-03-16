import os
from typing import Any, Dict

from pydantic import BaseModel

from agents.base_agent import BaseDevOpsAgent
from utils.azure_openai_client import AzureOpenAIClient

# Configuration class for the BuildPredictor agent
class BuildPredictorConfig(BaseModel):
    """Configuration settings for the Azure OpenAI build predictor."""

    azure_openai_endpoint: str = ""
    azure_openai_key: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-12-01-preview"

    @classmethod
    def from_env(cls) -> "BuildPredictorConfig":
        return cls(
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )

class BuildPredictorAgent(BaseDevOpsAgent):
    """Predict potential build failures using Azure OpenAI."""

    def __init__(self, config: BuildPredictorConfig):
        self.config = config
        self.client = AzureOpenAIClient(
            endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_key,
            deployment_name=config.azure_openai_deployment,
            api_version=config.azure_openai_api_version,
            temperature=0.2,
        )

    def predict_build_failure(self, build_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze build data and predict potential build failures.
        
        Args:
            build_data (Dict[str, Any]): Dictionary containing relevant build information
                such as commit data, previous build status, dependencies, etc.
        
        Returns:
            Dict[str, Any]: A dictionary containing:
                - prediction: The LLM's analysis and prediction
                - status: 'success' if prediction was generated, 'error' if an error occurred
                - error: Error message if status is 'error'
        """
        try:
            prediction = self.client.chat(
                system_prompt=(
                    "You are a build failure prediction assistant. Analyze build metadata and "
                    "explain failure risk, likely causes, and next checks."
                ),
                user_message=f"Build data: {build_data}",
            )
            return {"prediction": prediction, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}
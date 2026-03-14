from pydantic import BaseModel
from agents.base_agent import BaseDevOpsAgent
from groq import Groq
from typing import Dict, Any

# Configuration class for the BuildPredictor agent
class BuildPredictorConfig(BaseModel):
    """
    Configuration settings for the BuildPredictor agent.
    
    Attributes:
        model (str): The LLM model to be used for predictions (default: llama3-8b-8192)
        groq_api_key (str): API key for authentication with Groq's services
    """
    model: str = "llama3-8b-8192"  # Using Groq's recommended model
    groq_api_key: str

class BuildPredictorAgent(BaseDevOpsAgent):
    """
    An AI agent that predicts potential build failures by analyzing build data.
    
    This agent uses Groq's LLM to analyze build patterns and predict possible failures
    before they occur, enabling proactive issue resolution.
    """

    def __init__(self, config: BuildPredictorConfig):
        """
        Initialize the BuildPredictor agent.
        
        Args:
            config (BuildPredictorConfig): Configuration object containing model and API settings
        """
        self.config = config
        self.client = Groq(api_key=config.groq_api_key)

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
            # Create a chat completion request to analyze the build data
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a build failure prediction assistant. Analyze the build data and predict if the build might fail."
                    },
                    {
                        "role": "user",
                        "content": f"Please analyze this build data and predict if it might fail: {build_data}"
                    }
                ],
                model=self.config.model,
                temperature=0.7,  # Balance between creativity and consistency
                max_tokens=1024   # Maximum length of the generated response
            )
            
            return {
                "prediction": chat_completion.choices[0].message.content,
                "status": "success"
            }
        except Exception as e:
            # Return error information if the prediction fails
            return {"error": str(e), "status": "error"}
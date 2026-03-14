from pydantic import BaseModel
from agents.base_agent import BaseDevOpsAgent
import subprocess

class BuildStatusConfig(BaseModel):
    """
    Configuration settings for the BuildStatus agent.
    
    Attributes:
        image_tag (str): The Docker image tag to check for existence
    """
    image_tag: str

class BuildStatusAgent(BaseDevOpsAgent):
    """
    An agent that checks the build status of Docker images.
    
    This agent verifies whether a specified Docker image exists in the local Docker registry,
    which is useful for validating successful builds and deployments.
    """

    def __init__(self, config: BuildStatusConfig):
        """
        Initialize the BuildStatus agent.
        
        Args:
            config (BuildStatusConfig): Configuration object containing the image tag to check
        """
        self.config = config

    def check_build_status(self) -> str:
        """
        Check if a Docker image exists in the local registry.
        
        Uses Docker's inspect command to verify the existence of an image.
        If the image exists, the command returns a 0 exit code.
        
        Returns:
            str: A message indicating whether the image exists or an error message
                if the check fails
        """
        try:
            # Run docker inspect command to check if image exists
            result = subprocess.run(
                ["docker", "inspect", self.config.image_tag],
                stdout=subprocess.PIPE,    # Capture standard output
                stderr=subprocess.PIPE,    # Capture error output
                text=True                  # Return string instead of bytes
            )
            
            # Check the return code to determine if image exists
            if result.returncode == 0:
                return f"Docker image '{self.config.image_tag}' exists."
            else:
                return f"Docker image '{self.config.image_tag}' does not exist."
        except Exception as e:
            # Handle any errors that occur during the check
            return f"Error checking build status: {str(e)}"
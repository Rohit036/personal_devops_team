from pydantic import BaseModel
from agents.base_agent import BaseDevOpsAgent
from models.groq_models import DockerConfig
from utils.groq_client import GROQClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DockerfileConfig(BaseModel):
    """
    Configuration settings for the Dockerfile generator agent.
    
    Attributes:
        base_image (str): Base Docker image to use (e.g., nginx:alpine)
        expose_port (int): Port number to expose in the container
        copy_source (str): Source directory to copy into the container
        work_dir (str): Working directory inside the container
        groq_api_endpoint (str): GROQ API endpoint URL
        groq_api_key (str): Authentication key for GROQ API
    """
    base_image: str
    expose_port: int
    copy_source: str
    work_dir: str
    groq_api_endpoint: str
    groq_api_key: str


class DockerfileAgent(BaseDevOpsAgent):
    """
    An AI agent that generates and manages Dockerfile configurations.
    
    This agent can fetch configuration from GROQ's API and generate
    appropriate Dockerfile content based on the configuration.
    """

    def __init__(self, config: DockerfileConfig):
        """
        Initialize the Dockerfile agent with necessary configuration.
        
        Args:
            config (DockerfileConfig): Configuration object containing Docker and API settings
        """
        self.config = config
        self.groq_client = GROQClient(
            api_endpoint=config.groq_api_endpoint,
            api_key=config.groq_api_key
        )

    def fetch_config(self):
        """
        Fetch Dockerfile configuration from GROQ API.
        
        Queries the GROQ API for Docker configuration settings and updates
        the agent's configuration accordingly. Falls back to default values
        if the API request fails.
        """
        # Query GROQ API for Docker configuration
        groq_query = "*[_type == 'dockerConfig'][0]{baseImage, exposePort, copySource, workDir}"
        result = self.groq_client.query(groq_query)
        
        if result:
            # Update configuration with values from GROQ API
            self.config = DockerfileConfig(
                base_image=result.get("baseImage", "nginx:alpine"),
                expose_port=result.get("exposePort", 80),
                copy_source=result.get("copySource", "./html"),
                work_dir=result.get("workDir", "/usr/share/nginx/html"),
                groq_api_endpoint=result.get("groqApiEndpoint", ""),
                groq_api_key=result.get("groqApiKey", "")
            )
        else:
            # Fallback to default configuration if API request fails
            self.config = DockerfileConfig()

    def generate_dockerfile(self) -> str:
        """
        Generate Dockerfile content based on the current configuration.
        
        Returns:
            str: Complete Dockerfile content with appropriate instructions
                for building a container image
        """
        dockerfile = f"""
FROM {self.config.base_image}

WORKDIR {self.config.work_dir}

COPY {self.config.copy_source} .

EXPOSE {self.config.expose_port}

CMD ["nginx", "-g", "daemon off;"]
"""
        return dockerfile
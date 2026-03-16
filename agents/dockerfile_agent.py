from pydantic import BaseModel
from agents.base_agent import BaseDevOpsAgent

class DockerfileConfig(BaseModel):
    """
    Configuration settings for the Dockerfile generator agent.
    
    Attributes:
        base_image (str): Base Docker image to use (e.g., nginx:alpine)
        expose_port (int): Port number to expose in the container
        copy_source (str): Source directory to copy into the container
        work_dir (str): Working directory inside the container
    """
    base_image: str
    expose_port: int
    copy_source: str
    work_dir: str


class DockerfileAgent(BaseDevOpsAgent):
    """
    An AI agent that generates and manages Dockerfile configurations.
    
    This agent generates Dockerfile content based on the configuration.
    """

    def __init__(self, config: DockerfileConfig):
        """
        Initialize the Dockerfile agent with necessary configuration.
        
        Args:
            config (DockerfileConfig): Configuration object containing Docker and API settings
        """
        self.config = config

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
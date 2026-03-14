import os
from typing import Optional

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class AzureOpenAIClient:
    """
    Thin wrapper around ``langchain_openai.AzureChatOpenAI``.

    Configuration is read from constructor parameters first, falling back to
    the following environment variables:

    * ``AZURE_OPENAI_ENDPOINT``
    * ``AZURE_OPENAI_API_KEY``
    * ``AZURE_OPENAI_DEPLOYMENT_NAME``  (default: ``gpt-4o``)
    * ``AZURE_OPENAI_API_VERSION``       (default: ``2024-02-01``)
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.llm = AzureChatOpenAI(
            azure_endpoint=endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_deployment=deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=temperature,
        )

    def chat(self, system_prompt: str, user_message: str) -> str:
        """
        Send a system + user message pair and return the model's text reply.

        Args:
            system_prompt: Instructions / persona for the model.
            user_message:  The user's request.

        Returns:
            The model's response as a plain string.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = self.llm.invoke(messages)
        return response.content

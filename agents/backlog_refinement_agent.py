from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from agents.base_agent import BaseDevOpsAgent
from utils.azure_openai_client import AzureOpenAIClient


class BacklogRefinementConfig(BaseModel):
    """
    Configuration for the Backlog Refinement agent.

    Attributes:
        azure_openai_endpoint:    Azure OpenAI resource endpoint URL.
        azure_openai_key:         Azure OpenAI API key.
        azure_openai_deployment:  Deployment / model name (e.g. ``gpt-4o``).
        azure_openai_api_version: Azure OpenAI API version string.
        max_items_per_call:       Maximum backlog items to refine in a single
                                  LLM request (to stay within context limits).
    """

    azure_openai_endpoint: str = ""
    azure_openai_key: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-02-01"
    max_items_per_call: int = 10

    @classmethod
    def from_env(cls) -> "BacklogRefinementConfig":
        return cls(
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )


_SYSTEM_PROMPT = """You are an experienced Agile coach and product manager.
Your job is to refine raw backlog items into well-structured user stories.

For each item you receive, return a JSON array where every element has:
  - "original": the original item text
  - "user_story": a properly formatted "As a <role>, I want <goal>, so that <reason>" story
  - "acceptance_criteria": a list of clear, testable acceptance criteria (strings)
  - "story_points": an estimated effort in story points (1, 2, 3, 5, 8, 13)
  - "priority": one of "critical", "high", "medium", "low"
  - "labels": a list of relevant labels (e.g. ["backend", "api", "feature"])

Return only the JSON array — no extra text or markdown fences."""


class BacklogRefinementAgent(BaseDevOpsAgent):
    """
    An AI agent that refines raw backlog items into structured user stories.

    Uses Azure OpenAI (via :class:`~utils.azure_openai_client.AzureOpenAIClient`)
    to enrich each item with a proper user story format, acceptance criteria,
    story point estimates, and priority labels.
    """

    def __init__(self, config: Optional[BacklogRefinementConfig] = None):
        """
        Initialise the agent.

        Args:
            config: Agent configuration.  When *None*, values are read from
                    environment variables via :meth:`BacklogRefinementConfig.from_env`.
        """
        self.config = config or BacklogRefinementConfig.from_env()
        self.client = AzureOpenAIClient(
            endpoint=self.config.azure_openai_endpoint,
            api_key=self.config.azure_openai_key,
            deployment_name=self.config.azure_openai_deployment,
            api_version=self.config.azure_openai_api_version,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine_backlog(self, items: List[str]) -> Dict[str, Any]:
        """
        Refine a list of raw backlog items.

        Items are processed in batches (controlled by
        ``config.max_items_per_call``) to avoid hitting context-window limits.

        Args:
            items: Raw backlog items / feature descriptions.

        Returns:
            A dictionary with:
            - ``"status"``:  ``"success"`` or ``"error"``
            - ``"refined"``: list of refined item dicts (on success)
            - ``"error"``:   error message (on failure)
        """
        if not items:
            return {"status": "success", "refined": []}

        all_refined: List[Dict[str, Any]] = []
        batch_size = self.config.max_items_per_call

        try:
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                refined_batch = self._refine_batch(batch)
                all_refined.extend(refined_batch)

            return {"status": "success", "refined": all_refined}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def run(self) -> Dict[str, Any]:
        """
        Entry-point used by the LangGraph orchestrator.

        Returns an empty success result — callers should use
        :meth:`refine_backlog` directly when backlog items are available.
        """
        return {"status": "success", "refined": []}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refine_batch(self, batch: List[str]) -> List[Dict[str, Any]]:
        user_message = (
            "Please refine the following backlog items:\n\n"
            + "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(batch))
        )
        raw_response = self.client.chat(
            system_prompt=_SYSTEM_PROMPT,
            user_message=user_message,
        )
        # Strip optional markdown code fences that some models add
        cleaned = raw_response.strip()
        if cleaned.startswith("```") and "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0].strip()

        return json.loads(cleaned)

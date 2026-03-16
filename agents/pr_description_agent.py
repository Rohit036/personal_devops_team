import json
import os

from pydantic import BaseModel

from agents.base_agent import BaseDevOpsAgent
from utils.azure_openai_client import AzureOpenAIClient
from github import Github

_SYSTEM_PROMPT = """You are a senior software engineer writing pull request descriptions.
Given a PR title, the list of changed files with their diffs, and any existing description,
produce a clear, structured Markdown PR description that helps reviewers understand:
  - What changed and why (1-2 sentence summary)
  - A bullet list of the key changes per file/area
  - The type of change (bug fix, feature, refactor, docs, chore)
  - Concrete steps to test the changes
  - Any related issue numbers found in the title or diff (e.g. "Closes #42")

Return strict JSON with exactly these keys:
  summary        (string)          — 1-2 sentence plain-English summary
  changes        (array of string) — bullet points of what changed
  change_type    (string)          — one of: feature | bug_fix | refactor | docs | chore
  how_to_test    (array of string) — ordered test steps
  related_issues (array of string) — issue refs like "#42", empty array if none

Return JSON only, no markdown fences."""

# Chars of diff per file sent to the model — keeps token usage predictable.
_MAX_PATCH_CHARS = 1500
# Total chars across all file patches before we stop adding more files.
_MAX_TOTAL_PATCH_CHARS = 8000


class PRDescriptionConfig(BaseModel):
    azure_openai_endpoint: str = ""
    azure_openai_key: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-12-01-preview"
    github_token: str
    repo_name: str
    pull_request_number: int

    @classmethod
    def from_env(cls, repo_name: str, pull_request_number: int) -> "PRDescriptionConfig":
        return cls(
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            repo_name=repo_name,
            pull_request_number=pull_request_number,
        )


class PRDescriptionAgent(BaseDevOpsAgent):
    """
    Generates a structured PR description using Azure OpenAI and either
    updates the PR body (if it is blank/placeholder) or posts the suggestion
    as a comment so the developer's own description is never overwritten.
    """

    def __init__(self, config: PRDescriptionConfig):
        self.config = config
        self.azure_client = AzureOpenAIClient(
            endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_key,
            deployment_name=config.azure_openai_deployment,
            api_version=config.azure_openai_api_version,
            temperature=0.3,
        )
        self.github_client = Github(config.github_token)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _collect_diff_context(self, pr) -> str:
        """Build a compact diff summary from all changed files."""
        files = pr.get_files()
        parts: list[str] = []
        total = 0
        for f in files:
            patch = (f.patch or "").strip()
            if patch and len(patch) > _MAX_PATCH_CHARS:
                patch = patch[:_MAX_PATCH_CHARS] + "\n... (truncated)"
            entry = f"### {f.filename} (+{f.additions}/-{f.deletions})\n{patch}"
            if total + len(entry) > _MAX_TOTAL_PATCH_CHARS:
                parts.append(f"### {f.filename} (+{f.additions}/-{f.deletions})\n(diff omitted — token budget reached)")
                break
            parts.append(entry)
            total += len(entry)
        return "\n\n".join(parts)

    def _generate_description(self, title: str, existing_body: str, diff_context: str) -> dict:
        user_message = (
            f"PR Title: {title}\n\n"
            f"Existing description (may be empty):\n{existing_body or '(none)'}\n\n"
            f"Changed files and diffs:\n{diff_context}"
        )
        raw = self.azure_client.chat(
            system_prompt=_SYSTEM_PROMPT,
            user_message=user_message,
        )
        cleaned = raw.strip()
        if cleaned.startswith("```") and "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Unexpected response format from model")
        return parsed

    def _format_pr_body(self, data: dict) -> str:
        change_type_emoji = {
            "feature": " Feature",
            "bug_fix": " Bug Fix",
            "refactor": " Refactor",
            "docs": " Docs",
            "chore": " Chore",
        }
        change_label = change_type_emoji.get(data.get("change_type", ""), " Change")

        changes_md = "\n".join(f"- {c}" for c in data.get("changes", []))
        test_steps_md = "\n".join(f"{i+1}. {s}" for i, s in enumerate(data.get("how_to_test", [])))
        related = data.get("related_issues") or []
        related_md = ", ".join(related) if related else "_None identified_"

        return (
            f"## Summary\n\n"
            f"{data.get('summary', '')}\n\n"
            f"## Type of Change\n\n"
            f"{change_label}\n\n"
            f"## Changes Made\n\n"
            f"{changes_md}\n\n"
            f"## How to Test\n\n"
            f"{test_steps_md}\n\n"
            f"## Related Issues\n\n"
            f"{related_md}\n\n"
            f"---\n"
            f"_Description generated by the AI DevOps Team agent. Feel free to edit._"
        )

    @staticmethod
    def _is_placeholder(body: str | None) -> bool:
        """Return True if the PR body is empty or still has the GitHub default placeholder."""
        if not body or not body.strip():
            return True
        placeholder_hints = ["<!--", "describe your changes", "add a description"]
        lower = body.lower()
        return all(hint in lower for hint in ["<!--"]) or len(body.strip()) < 30

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        repo = self.github_client.get_repo(self.config.repo_name)
        pr = repo.get_pull(self.config.pull_request_number)

        diff_context = self._collect_diff_context(pr)
        generated = self._generate_description(
            title=pr.title,
            existing_body=pr.body,
            diff_context=diff_context,
        )
        formatted_body = self._format_pr_body(generated)

        if self._is_placeholder(pr.body):
            # PR has no real description — update it directly.
            pr.edit(body=formatted_body)
            action = "updated"
        else:
            # Developer already wrote a description — post as a suggestion comment.
            comment_body = (
                "##  Suggested PR Description\n\n"
                "The AI DevOps agent generated this description based on your diff. "
                "Copy it into the PR description if it looks useful.\n\n"
                "---\n\n"
                + formatted_body
            )
            pr.create_issue_comment(comment_body)
            action = "commented"

        return {
            "status": "success",
            "action": action,
            "pr_number": self.config.pull_request_number,
            "generated": generated,
        }

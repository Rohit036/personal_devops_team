import json
import os
import re
from typing import Any

from pydantic import BaseModel

from agents.base_agent import BaseDevOpsAgent
from utils.azure_openai_client import AzureOpenAIClient
from github import Github

_SYSTEM_PROMPT = """You are a security and dependency expert. Analyze the following new or updated packages in a pull request and identify:

1. Known security vulnerabilities or CVEs (if any are well-known)
2. Deprecated or unmaintained packages
3. License compatibility issues (MIT, GPL, proprietary clashes)
4. Major version bumps or breaking changes
5. Dependencies that appear suspicious or unusual for a DevOps project
6. Overall risk score for the PR (low/medium/high)

Format your response as strict JSON with exactly these keys:
  packages         (array) — array of {name, version, risks (array of strings), severity (low|medium|high)}
  overall_risk     (string) — one of: low | medium | high
  summary          (string) — 1-2 sentence summary of findings
  recommendations  (array) — actionable steps (e.g., "review CVE-2024-XXXXX before merge", "consider alternative")

Return JSON only, no markdown fences."""

_DEPENDENCY_FILES = {
    "requirements.txt": r"^\s*([a-zA-Z0-9\-_.]+)",  # Python: pkg==1.2.3
    "pyproject.toml": r'^\s*"?([a-zA-Z0-9\-_.]+)"?\s*=',  # Python: "pkg" = "1.2.3"
    "package.json": r'"([a-zA-Z0-9\-_./@]+)"\s*:',  # Node: "pkg": "1.2.3"
}

_MAX_DIFF_CHARS = 2000


class DependencyRiskConfig(BaseModel):
    azure_openai_endpoint: str = ""
    azure_openai_key: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-12-01-preview"
    github_token: str
    repo_name: str
    pull_request_number: int

    @classmethod
    def from_env(cls, repo_name: str, pull_request_number: int) -> "DependencyRiskConfig":
        return cls(
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            repo_name=repo_name,
            pull_request_number=pull_request_number,
        )


class DependencyRiskAgent(BaseDevOpsAgent):
    """
    Analyzes PR changes to dependency files (requirements.txt, package.json, etc.)
    and flags security/compatibility risks using Azure OpenAI.
    """

    def __init__(self, config: DependencyRiskConfig):
        self.config = config
        self.azure_client = AzureOpenAIClient(
            endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_key,
            deployment_name=config.azure_openai_deployment,
            api_version=config.azure_openai_api_version,
            temperature=0.2,
        )
        self.github_client = Github(config.github_token)

    def _is_dependency_file(self, filename: str) -> bool:
        """Check if a file is a known dependency manifest."""
        basename = filename.split("/")[-1]
        return basename in _DEPENDENCY_FILES

    def _extract_dependency_changes(self, pr_files: list) -> dict[str, str]:
        """Extract diffs from dependency files only."""
        changes = {}
        for file in pr_files:
            if self._is_dependency_file(file.filename):
                patch = (file.patch or "").strip()
                if patch:
                    if len(patch) > _MAX_DIFF_CHARS:
                        patch = patch[: _MAX_DIFF_CHARS] + "\n... (truncated)"
                    changes[file.filename] = patch
        return changes

    def _analyze_risks(self, dependency_changes: dict[str, str]) -> dict[str, Any]:
        """Send dependency changes to Azure OpenAI for risk assessment."""
        if not dependency_changes:
            return {"status": "no_changes", "overall_risk": "low", "summary": "No dependency files changed."}

        context = "PR Dependency Changes:\n\n"
        for filename, patch in dependency_changes.items():
            context += f"### {filename}\n```\n{patch}\n```\n\n"

        try:
            response = self.azure_client.chat(
                system_prompt=_SYSTEM_PROMPT,
                user_message=context,
            )
            cleaned = response.strip()
            if cleaned.startswith("```") and "\n" in cleaned:
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0].strip()
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError("Unexpected response format")
            return parsed
        except Exception as exc:
            return {"status": "error", "error": str(exc), "overall_risk": "unknown"}

    def _format_comment(self, analysis: dict[str, Any]) -> str:
        """Format risk analysis as a GitHub comment."""
        if analysis.get("status") == "no_changes":
            return "<!-- dependency-risk -->\n**✅ No dependency files changed in this PR.**"

        if analysis.get("status") == "error":
            return (
                f"<!-- dependency-risk -->\n"
                f"⚠️ **Dependency Risk Check Failed:**\n"
                f"{analysis.get('error', 'Unknown error')}"
            )

        risk = analysis.get("overall_risk", "unknown").upper()
        risk_emoji = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}[risk] if risk in risk_emoji else "❓"

        lines = [
            "<!-- dependency-risk -->",
            f"## {risk_emoji} Dependency Risk Assessment: {risk}",
            "",
            f"**Summary:** {analysis.get('summary', 'Analysis unavailable')}",
            "",
        ]

        packages = analysis.get("packages") or []
        if packages:
            lines.append("### Packages Analyzed")
            for pkg in packages:
                name = pkg.get("name", "?")
                version = pkg.get("version", "?")
                severity = pkg.get("severity", "unknown")
                risks = pkg.get("risks") or []

                lines.append(f"- **{name}** @ {version} [{severity}]")
                if risks:
                    for risk_item in risks:
                        lines.append(f"  - {risk_item}")
        else:
            lines.append("### Packages")
            lines.append("_No significant risks detected in added/updated packages._")

        recommendations = analysis.get("recommendations") or []
        if recommendations:
            lines.append("")
            lines.append("### Recommendations")
            for rec in recommendations:
                lines.append(f"- {rec}")

        lines += [
            "",
            "---",
            "_Analysis performed by AI Dependency Risk Agent._",
        ]
        return "\n".join(lines)

    def _upsert_comment(self, pr: Any, comment_body: str) -> None:
        """Post or update a single dependency risk comment on the PR."""
        marker = "<!-- dependency-risk -->"
        existing = None
        for comment in pr.get_issue_comments():
            if marker in (comment.body or ""):
                existing = comment
                break
        if existing:
            existing.edit(comment_body)
        else:
            pr.create_issue_comment(comment_body)

    def run(self) -> dict[str, Any]:
        """Execute the dependency risk analysis workflow."""
        repo = self.github_client.get_repo(self.config.repo_name)
        pr = repo.get_pull(self.config.pull_request_number)

        files = pr.get_files()
        dependency_changes = self._extract_dependency_changes(list(files))

        analysis = self._analyze_risks(dependency_changes)
        comment_body = self._format_comment(analysis)
        self._upsert_comment(pr, comment_body)

        return {
            "status": "success",
            "pr_number": self.config.pull_request_number,
            "risk_level": analysis.get("overall_risk", "unknown"),
            "files_analyzed": list(dependency_changes.keys()),
        }

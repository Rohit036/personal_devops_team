import argparse
import os
from typing import Any

from dotenv import load_dotenv
from github import Auth, Github

from orchestrator.langgraph_orchestrator import build_orchestrator

# Load environment variables from .env file
load_dotenv()

def main(
    mode: str = "quick",
    repo_name: str | None = None,
    max_issues: int = 5,
    issue_number: int | None = None,
):
    """Run local demos for backlog refinement workflows."""
    print("DevOps AI Team starting...")

    resolved_repo_name = repo_name or os.getenv("GITHUB_REPOSITORY", "owner/repo")

    if mode == "quick":
        print("\nQuick mode: refining sample backlog items")
        _run_orchestrator_demo(resolved_repo_name)
        print("\nQuick demo completed")
        return

    if mode == "issues":
        print("\nIssues mode: refining GitHub issues into stories")
        _run_github_issues_demo(
            resolved_repo_name,
            max_issues=max_issues,
            issue_number=issue_number,
        )
        print("\nIssues demo completed")
        return

    raise ValueError(f"Unsupported mode: {mode}. Use 'quick' or 'issues'.")


def _run_orchestrator_demo(repo_name: str):
    """
    Demonstrate the LangGraph orchestrator with a backlog-refinement task
    (safe to run without a real PR number).
    """
    sample_items = [
        "Add user authentication to the API",
        "Fix slow database queries on the reports page",
        "Create a dashboard for build metrics",
    ]

    print("  Refining sample backlog items...")
    _refine_backlog_items(repo_name, sample_items)


def _run_github_issues_demo(
    repo_name: str,
    max_issues: int = 5,
    issue_number: int | None = None,
):
    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        print("  GITHUB_TOKEN is missing; cannot read issues from GitHub.")
        return

    try:
        gh = Github(auth=Auth.Token(token))
        repo = gh.get_repo(repo_name)
        issue_backlog_items: list[str] = []
        issue_objects = []  # keep Issue objects for comment posting

        if issue_number is not None:
            issue = repo.get_issue(number=issue_number)
            if issue.pull_request is not None:
                print(f"  #{issue.number} is a pull request, not a backlog issue. Skipping.")
                return
            issue_backlog_items.append(_issue_to_backlog_item(issue.title, issue.body, issue.labels))
            issue_objects.append(issue)
        else:
            open_issues = repo.get_issues(state="open")
            for issue in open_issues:
                # GitHub API returns PRs in the issues list; skip those for backlog demo.
                if issue.pull_request is not None:
                    continue
                issue_backlog_items.append(_issue_to_backlog_item(issue.title, issue.body, issue.labels))
                issue_objects.append(issue)
                if len(issue_backlog_items) >= max_issues:
                    break

        if not issue_backlog_items:
            print(f"  No open issues found in {repo_name}.")
            return

        issue_refs = [f"#{i.number}" for i in issue_objects]
        print(
            f"  Loaded {len(issue_backlog_items)} open issue(s) from {repo_name}: "
            f"{', '.join(issue_refs)}"
        )
        refined = _refine_backlog_items(repo_name, issue_backlog_items)

        # Post the refined story back to each source GitHub issue as a comment.
        if refined:
            print("\n  Posting refined stories back to GitHub issues...")
            for issue, refined_item in zip(issue_objects, refined):
                _upsert_issue_story_comment(issue, refined_item)
                print(f"    Posted comment on #{issue.number}")
    except Exception as exc:
        print(f"  Failed to fetch issues from GitHub: {exc}")


def _format_comment(refined: dict) -> str:
    """Build a Markdown comment body to post on a GitHub issue."""
    user_story = refined.get("user_story") or refined.get("original", "")
    story_points = refined.get("story_points", "?")
    priority = refined.get("priority", "?")
    acceptance = refined.get("acceptance_criteria") or []
    labels = refined.get("labels") or []

    lines = [
        "## AI-Refined User Story",
        "",
        f"**User Story:** {user_story}",
        "",
        f"**Story Points:** {story_points}  |  **Priority:** {priority.capitalize()}",
    ]
    if labels:
        lines += ["", f"**Labels:** {', '.join(labels)}"]
    if acceptance:
        lines += ["", "**Acceptance Criteria:**"]
        for criterion in acceptance:
            lines.append(f"- {criterion}")
    lines += ["", "---", "_Generated automatically by the Personal DevOps Team AI Agent._"]
    return "\n".join(lines)


def _upsert_issue_story_comment(issue: Any, refined: dict) -> None:
    marker = "<!-- ai-backlog-refinement -->"
    comment_body = f"{marker}\n{_format_comment(refined)}"
    existing_comment = None
    for comment in issue.get_comments():
        if marker in (comment.body or ""):
            existing_comment = comment
            break
    if existing_comment:
        existing_comment.edit(comment_body)
    else:
        issue.create_comment(comment_body)


def _issue_to_backlog_item(title: str, body: str | None, labels) -> str:
    # Keep payload concise while preserving enough context for better refinement.
    compact_body = (body or "").strip().replace("\r\n", "\n")
    compact_body = " ".join(compact_body.split())
    if len(compact_body) > 500:
        compact_body = compact_body[:500] + "..."

    label_names = [label.name for label in labels] if labels else []
    labels_text = ", ".join(label_names) if label_names else "none"

    return f"Title: {title}\nDetails: {compact_body or 'No details provided'}\nLabels: {labels_text}"


def _refine_backlog_items(repo_name: str, backlog_items: list[str]) -> list[dict]:
    """Invoke the orchestrator for backlog refinement and print results. Returns the refined items."""
    graph = build_orchestrator()
    result = graph.invoke({
        "task": "backlog_refinement",
        "repo_name": repo_name,
        "pr_number": None,
        "backlog_items": backlog_items,
        "messages": [],
        "code_review_result": None,
        "backlog_result": None,
        "build_prediction": None,
        "build_status": None,
        "error": None,
    })

    for msg in result.get("messages", []):
        print(f"  {msg}")

    backlog_result = result.get("backlog_result", {})
    if backlog_result.get("status") == "success":
        refined = backlog_result.get("refined", [])
        for item in refined:
            print(f"\n  {item.get('user_story', item.get('original', ''))}")
            print(f"     Points: {item.get('story_points')}  Priority: {item.get('priority')}")
            acceptance = item.get("acceptance_criteria") or []
            if acceptance:
                print("     Acceptance Criteria:")
                for criterion in acceptance:
                    print(f"       - {criterion}")
            labels = item.get("labels") or []
            if labels:
                print(f"     Labels: {', '.join(labels)}")
        return refined
    else:
        print(f"    Backlog refinement skipped (Azure OpenAI not configured): "
              f"{backlog_result.get('error', 'unknown error')}")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DevOps AI Team demo")
    parser.add_argument(
        "--mode",
        choices=["quick", "issues"],
        default="quick",
        help="quick = static backlog demo, issues = GitHub issues backlog demo",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repo in owner/repo format (defaults to GITHUB_REPOSITORY env var)",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=5,
        help="Maximum number of open issues to fetch for issues mode",
    )
    parser.add_argument(
        "--issue-number",
        type=int,
        default=None,
        help="Process only one specific issue number in issues mode",
    )
    args = parser.parse_args()
    main(
        mode=args.mode,
        repo_name=args.repo,
        max_issues=args.max_issues,
        issue_number=args.issue_number,
    )
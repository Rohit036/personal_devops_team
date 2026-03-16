import argparse
import os
from typing import Any

from agents.build_predictor_agent import BuildPredictorAgent, BuildPredictorConfig
from agents.build_status_agent import BuildStatusAgent, BuildStatusConfig
from agents.dockerfile_agent import DockerfileAgent, DockerfileConfig
from agents.github_actions_agent import GitHubActionsAgent, GitHubActionsConfig
from orchestrator.langgraph_orchestrator import build_orchestrator
from dotenv import load_dotenv
from github import Auth, Github

# Load environment variables from .env file
load_dotenv()

def main(
    mode: str = "full",
    repo_name: str | None = None,
    max_issues: int = 5,
    issue_number: int | None = None,
):
    """
    Main orchestration function that coordinates the DevOps AI team's activities.
    
    This function manages five main tasks:
    1. Creating a GitHub Actions CI/CD pipeline
    2. Generating a Dockerfile
    3. Building and checking Docker image status
    4. Predicting build success/failure
    5. Demonstrating the LangGraph agent orchestrator
    """
    print("🤖 DevOps AI Team Starting Up...")

    resolved_repo_name = repo_name or os.getenv("GITHUB_REPOSITORY", "owner/repo")

    if mode == "quick":
        print("\n⚡ Quick mode: running only the Azure OpenAI backlog-refinement demo")
        _run_orchestrator_demo(resolved_repo_name)
        print("\n✨ Quick demo completed!")
        return

    if mode == "issues":
        print("\n🧾 Issues mode: reading open GitHub issues and refining them as stories")
        _run_github_issues_demo(
            resolved_repo_name,
            max_issues=max_issues,
            issue_number=issue_number,
        )
        print("\n✨ Issues demo completed!")
        return

    # 1. Create GitHub Actions Pipeline
    print("\n1️⃣ GitHub Actions Agent: Creating CI/CD Pipeline...")
    gha_config = GitHubActionsConfig(
        workflow_name="CI Pipeline",
        python_version="3.13.0",
        run_tests=True,
    )
    gha_agent = GitHubActionsAgent(config=gha_config)
    pipeline = gha_agent.generate_pipeline()
    
    # Save the pipeline configuration to a YAML file
    with open(".github/workflows/CI3.yml", "w") as f:
        f.write(pipeline)
    print("✅ CI/CD Pipeline created!")

    # 2. Create Dockerfile
    print("\n2️⃣ Dockerfile Agent: Creating Dockerfile...")
    docker_config = DockerfileConfig(
        base_image="nginx:alpine",        # Using lightweight nginx image
        expose_port=80,                   # Standard HTTP port
        copy_source="./html",             # Source directory for web content
        work_dir="/usr/share/nginx/html", # Default nginx content directory
    )
    docker_agent = DockerfileAgent(config=docker_config)
    dockerfile = docker_agent.generate_dockerfile()
    
    # Save the Dockerfile
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)
    print("✅ Dockerfile created!")

    # 3. Build and Check Status
    print("\n3️⃣ Build Status Agent: Building and checking Docker image...")
    status_config = BuildStatusConfig(image_tag="myapp:latest")
    status_agent = BuildStatusAgent(config=status_config)
    
    # Attempt to build the Docker image
    print("🔨 Building Docker image...")
    import subprocess
    build_result = subprocess.run(
        ["docker", "build", "-t", "myapp:latest", "."], 
        capture_output=True,  # Capture command output
        text=True            # Return string instead of bytes
    )
    
    # Verify the build status
    status = status_agent.check_build_status()
    print(f"📊 Build Status: {status}")

    # 4. Predict Build Success/Failure
    print("\n4️⃣ Build Predictor Agent: Analyzing build patterns...")
    predictor_config = BuildPredictorConfig(
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        azure_openai_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )
    predictor_agent = BuildPredictorAgent(config=predictor_config)
    
    # Prepare build data for analysis
    build_data = {
        "dockerfile_exists": True,         # Dockerfile was created
        "ci_pipeline_exists": True,        # CI pipeline was created
        "last_build_status": status,       # Result of the latest build
        "python_version": "3.13.0",        # Python version being used
        "dependencies_updated": True       # Dependencies are current
    }
    
    # Get build prediction
    prediction = predictor_agent.predict_build_failure(build_data)
    print(f"🔮 Build Prediction: {prediction}")

    # 5. LangGraph Orchestrator Demo
    print("\n5️⃣ LangGraph Orchestrator: Running full-pipeline demo...")
    _run_orchestrator_demo(resolved_repo_name)

    print("\n✨ DevOps AI Team has completed their tasks!")


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

    print("  📋 Refining sample backlog items with Azure OpenAI + LangGraph...")
    _refine_backlog_items(repo_name, sample_items)


def _run_github_issues_demo(
    repo_name: str,
    max_issues: int = 5,
    issue_number: int | None = None,
):
    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        print("  ⚠️  GITHUB_TOKEN is missing; cannot read issues from GitHub.")
        return

    try:
        gh = Github(auth=Auth.Token(token))
        repo = gh.get_repo(repo_name)
        issue_backlog_items: list[str] = []
        issue_objects = []  # keep Issue objects for comment posting

        if issue_number is not None:
            issue = repo.get_issue(number=issue_number)
            if issue.pull_request is not None:
                print(f"  ⚠️  #{issue.number} is a pull request, not a backlog issue. Skipping.")
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
            print(f"  ⚠️  No open issues found in {repo_name}.")
            return

        issue_refs = [f"#{i.number}" for i in issue_objects]
        print(
            f"  📥 Loaded {len(issue_backlog_items)} open issue(s) from {repo_name}: "
            f"{', '.join(issue_refs)}"
        )
        refined = _refine_backlog_items(repo_name, issue_backlog_items)

        # Post the refined story back to each source GitHub issue as a comment.
        if refined:
            print("\n  💬 Posting refined stories back to GitHub issues...")
            for issue, refined_item in zip(issue_objects, refined):
                _upsert_issue_story_comment(issue, refined_item)
                print(f"     ✅ Posted comment on #{issue.number}")
    except Exception as exc:
        print(f"  ❌ Failed to fetch issues from GitHub: {exc}")


def _format_comment(refined: dict) -> str:
    """Build a Markdown comment body to post on a GitHub issue."""
    user_story = refined.get("user_story") or refined.get("original", "")
    story_points = refined.get("story_points", "?")
    priority = refined.get("priority", "?")
    acceptance = refined.get("acceptance_criteria") or []
    labels = refined.get("labels") or []

    lines = [
        "## 🤖 AI-Refined User Story",
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
            print(f"\n  📌 {item.get('user_story', item.get('original', ''))}")
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
        print(f"  ⚠️  Backlog refinement skipped (Azure OpenAI not configured): "
              f"{backlog_result.get('error', 'unknown error')}")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DevOps AI Team demo")
    parser.add_argument(
        "--mode",
        choices=["quick", "issues", "full"],
        default="full",
        help="quick = static backlog demo, issues = GitHub issues backlog demo, full = complete pipeline",
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
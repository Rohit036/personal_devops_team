# Personal DevOps Team

Lean AI-agent demo for engineering workflows on GitHub.

This repo is intentionally focused on must-have value:
- PR description generation
- Dependency risk analysis
- PR code review + build risk prediction
- Issue-to-story backlog refinement with acceptance criteria

All AI calls use Azure OpenAI.

## What is included

- PR workflow: `.github/workflows/pr_agents.yml`
  - On PR opened: generate/update PR description
  - On PR synchronize/reopened: run code review + build prediction
  - On every PR event: run dependency risk analyzer and upsert one comment

- Backlog workflow: `.github/workflows/backlog_agents.yml`
  - On issue opened/reopened: refine that issue into a user story
  - Upserts one issue comment with acceptance criteria, story points, priority, labels

- Local CLI demo: `main.py`
  - `--mode quick`: refine sample backlog items
  - `--mode issues`: fetch issue(s), refine, and post/update comments

## Architecture

- `agents/pr_description_agent.py`
- `agents/dependency_risk_agent.py`
- `agents/code_review_agent.py`
- `agents/backlog_refinement_agent.py`
- `agents/build_predictor_agent.py`
- `orchestrator/langgraph_orchestrator.py`
- `utils/azure_openai_client.py`

## Setup

1. Create `.env` from `dot_env_example`

```bash
cp dot_env_example .env
```

2. Fill required variables in `.env`

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview

GITHUB_TOKEN=your_github_token
GITHUB_REPOSITORY=owner/repo
```

3. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Local demo commands

Quick backlog refinement:

```bash
python main.py --mode quick
```

Refine open issues from a repo:

```bash
python main.py --mode issues --repo owner/repo --max-issues 5
```

Refine one specific issue (used by issue workflow):

```bash
python main.py --mode issues --repo owner/repo --issue-number 123 --max-issues 1
```

## GitHub Actions demo

1. Push a PR that changes code and optionally dependency files
2. Open PR and watch `PR Agents Pipeline`
3. You should see:
   - PR description generated/updated
   - Dependency risk comment upserted
   - On subsequent pushes: code review + build prediction updates

For backlog demo:
1. Open an issue
2. Watch `Backlog Agents Pipeline`
3. A refined story comment is added/updated on that issue

## Notes

- The repository is optimized for demo clarity over feature breadth.
- Legacy Docker/static-site scaffolding was intentionally removed.

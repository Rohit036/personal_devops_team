# TALKITDOIT - DevOps AI Agent Team 🤖

Welcome to the talkitdoit project! This repository contains a team of AI agents that help automate and enhance your DevOps workflow. As featured on our [YouTube Channel](youtube.com/@talkitdoit), these agents work together to handle various DevOps tasks including code review, backlog refinement, build prediction, and infrastructure management.

The agents are now orchestrated by a **LangGraph** state machine and backed by **Azure OpenAI**, making this repo a practical demo for:
- GitHub + Azure OpenAI + LangGraph experimentation
- Rapid delivery and developer experience patterns
- AI agent design standards and orchestration

- s an engineering manager, I want a release health dashboard to monitor deployment reliability, so that I can ensure stable releases and address issues proactively.
     Points: 13  Priority: critical
     Acceptance Criteria:
       - Dashboard displays deployment success rate, failed deployment count, mean time to recovery, and rollback frequency.
       - Daily and weekly trend views are available for all tracked metrics.
       - Filters are available for different environments (dev, staging, prod).
       - Users can export deployment metrics to CSV.
       - Access to the dashboard is restricted to engineering managers.
     Labels: frontend, dashboard, analytics, enhancement

  💬 Posting refined stories back to GitHub issues...
     ✅ Posted comment on #5
     ✅ Posted comment on #4

[![YouTube Channel](https://img.shields.io/badge/YouTube-Subscribe-red)](https://www.youtube.com/@talkitdoit)
[![GitHub Stars](https://img.shields.io/github/stars/talkitdoit/talkitdoit-ai?style=social)](https://github.com/talkitdoit/build-a-devops-team-using-ai-agents)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🌟 Features

- 🔄 Automated CI/CD Pipeline Generation
- 🐳 Docker Configuration Management
- 📊 Build Success Prediction (Azure OpenAI)
- 🔍 AI-Powered Code Review (posts directly to GitHub PRs)
- 💬 Natural Language PR Interaction
- 📈 Real-time Build Status Monitoring
- 📋 **Backlog Refinement Agent** — turns raw ideas into structured user stories (Azure OpenAI)
- 🧠 **LangGraph Orchestrator** — state-machine coordination of all agents
- ☁️ **Azure OpenAI** support (GPT-4o / GPT-4 Turbo)

## 🏗️ Architecture

```
main.py
  └── LangGraph Orchestrator  (orchestrator/langgraph_orchestrator.py)
        ├── code_review        → agents/code_review_agent.py      (Azure OpenAI)
        ├── backlog_refinement → agents/backlog_refinement_agent.py (Azure OpenAI)
        └── build_prediction   → agents/build_predictor_agent.py   (Azure OpenAI)

Standalone agents (also callable directly):
  agents/github_actions_agent.py   — generates CI/CD workflow YAML
  agents/dockerfile_agent.py       — generates Dockerfiles
  agents/build_status_agent.py     — checks local Docker image status
  agents/chat_agent.py             — general PR chat assistant
```

## 🚀 Prerequisites & Assumptions

### Required Accounts

| Account | Purpose | Free Tier |
|---------|---------|-----------|
| [GitHub](https://github.com/signup) | Repo hosting + CI/CD | ✅ unlimited public repos |
| [Azure](https://azure.microsoft.com/free/) | Azure OpenAI (GPT-4o) | ✅ $200 credit for new accounts |

### Technical Requirements
- Python 3.13.0 or higher
- Docker Desktop
- Git

### Setting Up GitHub Secrets

Go to **Settings → Secrets and variables → Actions** and add:

```
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION="2024-12-01-preview"

# GitHub
GH_TOKEN=your_github_personal_access_token
```

To create a GitHub Personal Access Token:
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Click "Generate new token (classic)"
3. Select `repo` and `workflow` permissions

## 🚀 Getting Started

### Installation

#### macOS

```bash
brew install python@3.13
brew install --cask docker

git clone https://github.com/Rohit036/personal_devops_team.git
cd personal_devops_team

python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Windows

```powershell
# Install Python 3.13 from https://www.python.org/downloads/
# Install Docker Desktop from https://www.docker.com/products/docker-desktop

git clone https://github.com/Rohit036/personal_devops_team.git
cd personal_devops_team

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

#### Linux

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13 python3.13-venv
sudo apt install docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER

git clone https://github.com/Rohit036/personal_devops_team.git
cd personal_devops_team

python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Copy the example file and fill in your keys:

```bash
cp dot_env_example .env
# Edit .env and add your API keys
```

Required variables:

```bash
# Azure OpenAI (for all AI agents + orchestrator)
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_API_VERSION="2024-12-01-preview"

# GitHub
GITHUB_TOKEN=your_github_token
```

### Usage

```bash
source venv/bin/activate   # macOS/Linux
# .\venv\Scripts\activate  # Windows

# Quick mode (recommended first run; no Docker build required)
python main.py --mode quick

# Full mode (runs CI/Docker generation + docker build + predictor + orchestrator)
python main.py --mode full
```

### Using the LangGraph Orchestrator directly

```python
from orchestrator.langgraph_orchestrator import build_orchestrator

graph = build_orchestrator()

# Refine backlog items
result = graph.invoke({
    "task": "backlog_refinement",
    "repo_name": "owner/repo",
    "pr_number": None,
    "backlog_items": [
        "Add OAuth2 login",
        "Fix memory leak in worker service",
        "Build deployment dashboard",
    ],
    "messages": [],
    "code_review_result": None,
    "backlog_result": None,
    "build_prediction": None,
    "build_status": None,
    "error": None,
})

for item in result["backlog_result"]["refined"]:
    print(item["user_story"])
    print("  Acceptance criteria:", item["acceptance_criteria"])
    print("  Story points:", item["story_points"])

# Code review on a PR
result = graph.invoke({
    "task": "code_review",
    "repo_name": "owner/repo",
    "pr_number": 42,
    "backlog_items": None,
    "messages": [],
    "code_review_result": None,
    "backlog_result": None,
    "build_prediction": None,
    "build_status": None,
    "error": None,
})
```

### Project Structure

```
personal_devops_team/
├── agents/
│   ├── base_agent.py              # Base class for all agents
│   ├── backlog_refinement_agent.py # NEW – Azure OpenAI backlog agent
│   ├── code_review_agent.py        # PR code review (Azure OpenAI)
│   ├── chat_agent.py               # PR chat assistant (Azure OpenAI)
│   ├── build_predictor_agent.py    # Build failure prediction (Azure OpenAI)
│   ├── build_status_agent.py       # Docker image status check
│   ├── dockerfile_agent.py         # Dockerfile generator
│   └── github_actions_agent.py     # CI/CD workflow generator
├── orchestrator/
│   └── langgraph_orchestrator.py   # NEW – LangGraph state machine
├── utils/
│   ├── azure_openai_client.py      # NEW – Azure OpenAI wrapper
├── html/                           # Web interface files
├── .github/workflows/
│   └── ai_agents.yml               # GitHub Actions pipeline
├── main.py                         # Main orchestration script
├── requirements.txt                # Python dependencies
└── dot_env_example                 # Example environment config
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-agent`)
3. Commit your changes (`git commit -m 'Add new agent'`)
4. Push to the branch (`git push origin feature/my-agent`)
5. Open a Pull Request — the AI agents will review it automatically!

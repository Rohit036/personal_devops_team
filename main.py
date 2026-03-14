from agents.github_actions_agent import GitHubActionsAgent, GitHubActionsConfig
from agents.dockerfile_agent import DockerfileAgent, DockerfileConfig
from agents.build_predictor_agent import BuildPredictorAgent, BuildPredictorConfig
from agents.build_status_agent import BuildStatusAgent, BuildStatusConfig
from orchestrator.langgraph_orchestrator import build_orchestrator
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
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

    # 1. Create GitHub Actions Pipeline
    print("\n1️⃣ GitHub Actions Agent: Creating CI/CD Pipeline...")
    gha_config = GitHubActionsConfig(
        workflow_name="CI Pipeline",
        python_version="3.13.0",
        run_tests=True,
        groq_api_endpoint=os.getenv("GROQ_API_ENDPOINT"),
        groq_api_key=os.getenv("GROQ_API_KEY")
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
        groq_api_endpoint=os.getenv("GROQ_API_ENDPOINT"),
        groq_api_key=os.getenv("GROQ_API_KEY")
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
        model="llama3-8b-8192",  # Using Groq's recommended model
        groq_api_key=os.getenv("GROQ_API_KEY")
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
    repo_name = os.getenv("GITHUB_REPOSITORY", "owner/repo")
    _run_orchestrator_demo(repo_name)

    print("\n✨ DevOps AI Team has completed their tasks!")


def _run_orchestrator_demo(repo_name: str):
    """
    Demonstrate the LangGraph orchestrator with a backlog-refinement task
    (safe to run without a real PR number).
    """
    graph = build_orchestrator()

    sample_items = [
        "Add user authentication to the API",
        "Fix slow database queries on the reports page",
        "Create a dashboard for build metrics",
    ]

    print("  📋 Refining sample backlog items with Azure OpenAI + LangGraph...")
    result = graph.invoke({
        "task": "backlog_refinement",
        "repo_name": repo_name,
        "pr_number": None,
        "backlog_items": sample_items,
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
        for item in backlog_result.get("refined", []):
            print(f"\n  📌 {item.get('user_story', item.get('original', ''))}")
            print(f"     Points: {item.get('story_points')}  Priority: {item.get('priority')}")
    else:
        print(f"  ⚠️  Backlog refinement skipped (Azure OpenAI not configured): "
              f"{backlog_result.get('error', 'unknown error')}")


if __name__ == "__main__":
    main()
"""LangGraph orchestrator for repository automation tasks.

Supported tasks:
- code_review: run PR code review only
- backlog_refinement: transform raw backlog items into user stories
- build_prediction: estimate build risk from PR metadata
- full_pipeline: run code_review and then build_prediction
"""

from __future__ import annotations

import operator
import os
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict


TASK_CODE_REVIEW = "code_review"
TASK_BACKLOG_REFINEMENT = "backlog_refinement"
TASK_BUILD_PREDICTION = "build_prediction"
TASK_FULL_PIPELINE = "full_pipeline"


# ---------------------------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------------------------

class DevOpsAgentState(TypedDict):
    """
    Mutable state passed between nodes in the DevOps LangGraph pipeline.

    All fields are optional so that individual nodes only need to populate
    the fields they produce.
    """

    # --- inputs ---
    task: str
    """One of: ``code_review``, ``backlog_refinement``, ``build_prediction``,
    ``full_pipeline``."""

    repo_name: str
    """GitHub repository in ``owner/repo`` format."""

    pr_number: Optional[int]
    """Pull-request number (required for code-review tasks)."""

    backlog_items: Optional[List[str]]
    """Raw backlog items to refine (required for backlog-refinement tasks)."""

    # --- outputs ---
    messages: Annotated[List[str], operator.add]
    """Append-only log of messages produced by each node.

    LangGraph uses the ``operator.add`` reducer so that each node's messages
    are appended to the list rather than replacing it.
    """

    code_review_result: Optional[Dict[str, Any]]
    backlog_result: Optional[Dict[str, Any]]
    build_prediction: Optional[Dict[str, Any]]
    build_status: Optional[str]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def _run_code_review_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Run code review for the PR in state and append a status message."""
    from agents.code_review_agent import CodeReviewAgent, CodeReviewConfig

    pr_number = state.get("pr_number")
    pr_number_int = int(pr_number) if pr_number is not None else 0
    try:
        config = CodeReviewConfig(
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            repo_name=state.get("repo_name", ""),
            pull_request_number=pr_number_int,
        )
        agent = CodeReviewAgent(config=config)
        feedback = agent.run()
        return {
            **state,
            "code_review_result": {"status": "success", "feedback": feedback},
            "messages": [f"Code review completed: {len(feedback)} file(s) reviewed"],
        }
    except Exception as exc:
        return {
            **state,
            "code_review_result": {"status": "error", "error": str(exc)},
            "messages": [f"Code review error: {exc}"],
        }


def _run_backlog_refinement_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Run backlog refinement for provided backlog_items and append status."""
    from agents.backlog_refinement_agent import BacklogRefinementAgent, BacklogRefinementConfig

    items: List[str] = state.get("backlog_items") or []
    try:
        config = BacklogRefinementConfig.from_env()
        agent = BacklogRefinementAgent(config=config)
        result = agent.refine_backlog(items)
        count = len(result.get("refined", []))
        return {
            **state,
            "backlog_result": result,
            "messages": [f"Backlog refinement completed: {count} item(s) refined"],
        }
    except Exception as exc:
        return {
            **state,
            "backlog_result": {"status": "error", "error": str(exc)},
            "messages": [f"Backlog refinement error: {exc}"],
        }


def _run_build_prediction_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Run build risk prediction and append a status message."""
    from agents.build_predictor_agent import BuildPredictorAgent, BuildPredictorConfig

    try:
        config = BuildPredictorConfig(
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        agent = BuildPredictorAgent(config=config)
        build_data: Dict[str, Any] = {
            "repo_name": state.get("repo_name", ""),
            "pr_number": state.get("pr_number", 0),
            "code_review_status": (state.get("code_review_result") or {}).get("status", "not_run"),
        }
        result = agent.predict_build_failure(build_data)
        return {
            **state,
            "build_prediction": result,
            "messages": [f"Build prediction: {result.get('status', 'unknown')}"],
        }
    except Exception as exc:
        return {
            **state,
            "build_prediction": {"status": "error", "error": str(exc)},
            "messages": [f"Build prediction error: {exc}"],
        }


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _route_entry(state: DevOpsAgentState) -> str:
    """Select the graph entry node from the task name."""
    task = state.get("task", "code_review")
    routing: Dict[str, str] = {
        TASK_CODE_REVIEW: TASK_CODE_REVIEW,
        TASK_BACKLOG_REFINEMENT: TASK_BACKLOG_REFINEMENT,
        TASK_BUILD_PREDICTION: TASK_BUILD_PREDICTION,
        TASK_FULL_PIPELINE: TASK_CODE_REVIEW,
    }
    return routing.get(task, END)


def _route_after_code_review(state: DevOpsAgentState) -> str:
    """After code review, continue only when task requests full pipeline."""
    if state.get("task") == TASK_FULL_PIPELINE:
        return TASK_BUILD_PREDICTION
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_orchestrator():
    """Build and compile the orchestrator graph used by CLI and workflows."""
    builder = StateGraph(DevOpsAgentState)

    # Register nodes
    builder.add_node(TASK_CODE_REVIEW, _run_code_review_node)
    builder.add_node(TASK_BACKLOG_REFINEMENT, _run_backlog_refinement_node)
    builder.add_node(TASK_BUILD_PREDICTION, _run_build_prediction_node)

    # Conditional entry point routes to the right starting node
    builder.set_conditional_entry_point(
        _route_entry,
        {
            TASK_CODE_REVIEW: TASK_CODE_REVIEW,
            TASK_BACKLOG_REFINEMENT: TASK_BACKLOG_REFINEMENT,
            TASK_BUILD_PREDICTION: TASK_BUILD_PREDICTION,
            END: END,
        },
    )

    # After code review: either stop or continue to build prediction
    builder.add_conditional_edges(
        TASK_CODE_REVIEW,
        _route_after_code_review,
        {TASK_BUILD_PREDICTION: TASK_BUILD_PREDICTION, END: END},
    )

    # Terminal nodes go straight to END
    builder.add_edge(TASK_BACKLOG_REFINEMENT, END)
    builder.add_edge(TASK_BUILD_PREDICTION, END)

    return builder.compile()

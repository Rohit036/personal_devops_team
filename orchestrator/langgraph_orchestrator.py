"""
LangGraph-based orchestrator for the Personal DevOps AI Team.

The orchestrator models the agent pipeline as a directed state graph:

    ┌──────────────────────────────────────────┐
    │          Conditional entry point          │
    │  (routes to the correct starting node     │
    │   based on ``state["task"]``)             │
    └──┬──────────────┬──────────────┬──────────┘
       │              │              │
       ▼              ▼              ▼
  code_review   backlog_refine  build_prediction
       │              │              │
       ▼              ▼              ▼
    (after code review, full_pipeline continues
     to build_prediction; otherwise goes to END)

Supported tasks
---------------
* ``"code_review"``       — run the code-review agent on a pull request.
* ``"backlog_refinement"``— refine raw backlog items with the backlog agent.
* ``"build_prediction"``  — predict whether a build will succeed.
* ``"full_pipeline"``     — run code_review → build_prediction in sequence.
"""

from __future__ import annotations

import operator
import os
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict


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
    """Invoke the CodeReviewAgent and store its result in the shared state."""
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
            "messages": [f" Code review completed: {len(feedback)} file(s) reviewed"],
        }
    except Exception as exc:
        return {
            **state,
            "code_review_result": {"status": "error", "error": str(exc)},
            "messages": [f" Code review error: {exc}"],
        }


def _run_backlog_refinement_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Invoke the BacklogRefinementAgent and store its result in the shared state."""
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
            "messages": [f" Backlog refinement completed: {count} item(s) refined"],
        }
    except Exception as exc:
        return {
            **state,
            "backlog_result": {"status": "error", "error": str(exc)},
            "messages": [f" Backlog refinement error: {exc}"],
        }


def _run_build_prediction_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Invoke the BuildPredictorAgent and store its result in the shared state."""
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
            "messages": [f" Build prediction: {result.get('status', 'unknown')}"],
        }
    except Exception as exc:
        return {
            **state,
            "build_prediction": {"status": "error", "error": str(exc)},
            "messages": [f" Build prediction error: {exc}"],
        }


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _route_entry(state: DevOpsAgentState) -> str:
    """
    Conditional entry-point: select the first node to run based on
    ``state["task"]``.
    """
    task = state.get("task", "code_review")
    routing: Dict[str, str] = {
        "code_review": "code_review",
        "backlog_refinement": "backlog_refinement",
        "build_prediction": "build_prediction",
        "full_pipeline": "code_review",
    }
    return routing.get(task, END)


def _route_after_code_review(state: DevOpsAgentState) -> str:
    """After code review, continue to build prediction only for full_pipeline."""
    if state.get("task") == "full_pipeline":
        return "build_prediction"
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_orchestrator():
    """
    Compile and return the LangGraph ``CompiledGraph`` for the DevOps AI team.

    Example usage::

        graph = build_orchestrator()
        result = graph.invoke({
            "task": "code_review",
            "repo_name": "owner/repo",
            "pr_number": 42,
            "messages": [],
        })
        print(result["code_review_result"])
    """
    builder = StateGraph(DevOpsAgentState)

    # Register nodes
    builder.add_node("code_review", _run_code_review_node)
    builder.add_node("backlog_refinement", _run_backlog_refinement_node)
    builder.add_node("build_prediction", _run_build_prediction_node)

    # Conditional entry point routes to the right starting node
    builder.set_conditional_entry_point(
        _route_entry,
        {
            "code_review": "code_review",
            "backlog_refinement": "backlog_refinement",
            "build_prediction": "build_prediction",
            END: END,
        },
    )

    # After code review: either stop or continue to build prediction
    builder.add_conditional_edges(
        "code_review",
        _route_after_code_review,
        {"build_prediction": "build_prediction", END: END},
    )

    # Terminal nodes go straight to END
    builder.add_edge("backlog_refinement", END)
    builder.add_edge("build_prediction", END)

    return builder.compile()

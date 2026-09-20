# ============================================================
# backend/langgraph_workflow.py - 向后兼容重导出
# 实际逻辑已拆分到 backend/workflow/ (nodes.py + graph.py)
# ============================================================

from backend.workflow.graph import run_crisis_analysis, build_workflow, WorkflowState

__all__ = ["run_crisis_analysis", "build_workflow", "WorkflowState"]

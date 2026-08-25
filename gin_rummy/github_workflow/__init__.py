"""
GitHub MCP Autonomous Workflow Automation Package.
"""

from gin_rummy.github_workflow.mcp_automation import (
    GitHubMCPClient,
    GitHubWorkflowManager,
    MilestoneIssueDefinition,
    MILESTONE_DEFINITIONS,
)

__all__ = [
    "GitHubMCPClient",
    "GitHubWorkflowManager",
    "MilestoneIssueDefinition",
    "MILESTONE_DEFINITIONS",
]

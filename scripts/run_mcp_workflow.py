#!/usr/bin/env python3
"""
CLI Runner for Autonomous GitHub MCP Workflow.

Executes repository provisioning, milestone issue creation, feature branching,
code commits, pull requests, automated review, and merging via GitHub MCP tools.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gin_rummy.github_workflow.mcp_automation import (
    GitHubMCPClient,
    GitHubWorkflowManager,
    MILESTONE_DEFINITIONS,
)


def collect_scaffolding_files(root_dir: Path) -> List[Dict[str, str]]:
    """Collect core scaffolding files to push to GitHub."""
    target_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "pyproject.toml",
        "setup.py",
        ".github/workflows/ci.yml",
        "gin_rummy/__init__.py",
        "gin_rummy/github_workflow/__init__.py",
        "gin_rummy/github_workflow/mcp_automation.py",
        "scripts/run_mcp_workflow.py",
    ]

    files_payload: List[Dict[str, str]] = []
    for rel_path in target_files:
        full_path = root_dir / rel_path
        if full_path.exists() and full_path.is_file():
            try:
                content = full_path.read_text(encoding="utf-8")
                # Normalize path separators for git
                git_path = rel_path.replace("\\", "/")
                files_payload.append({"path": git_path, "content": content})
            except Exception as e:
                logging.warning(f"Could not read {rel_path}: {e}")

    return files_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous GitHub MCP Workflow Execution Script"
    )
    parser.add_argument(
        "--owner",
        type=str,
        default="mrriadh9-boop",
        help="GitHub repository owner / organization (default: mrriadh9-boop)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="rlcard-gin-rummy-bot",
        help="GitHub repository name (default: rlcard-gin-rummy-bot)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in simulated mode without mutating remote GitHub state",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["all", "setup-repo", "create-issues", "feature-lifecycle", "status"],
        default="all",
        help="Workflow action to execute (default: all)",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="feature/m1-scaffolding-mcp-setup",
        help="Feature branch name for lifecycle execution",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 70)
    print(" GitHub MCP Autonomous Workflow Runner")
    print(f" Target Repository : {args.owner}/{args.repo}")
    print(f" Execution Mode    : {'DRY-RUN (Simulated)' if args.dry_run else 'LIVE MCP'}")
    print(f" Selected Action   : {args.action}")
    print("=" * 70)

    client = GitHubMCPClient(
        owner=args.owner,
        repo=args.repo,
        dry_run=args.dry_run,
    )
    manager = GitHubWorkflowManager(client)

    scaffolding_files = collect_scaffolding_files(PROJECT_ROOT)
    logging.info(f"Collected {len(scaffolding_files)} repository files to sync.")

    try:
        if args.action == "setup-repo":
            print("\n[Step 1/1] Setting up repository...")
            res = manager.setup_repository()
            print(f"Result: {res}")

        elif args.action == "create-issues":
            print(f"\n[Step 1/1] Creating {len(MILESTONE_DEFINITIONS)} milestone issues...")
            res = manager.create_milestone_issues()
            print(f"Created {len(res)} issues successfully.")

        elif args.action == "feature-lifecycle":
            print(f"\n[Step 1/1] Executing feature lifecycle on branch '{args.branch}'...")
            res = manager.execute_feature_lifecycle(
                branch_name=args.branch,
                files_to_push=scaffolding_files,
                commit_message="feat(scaffolding): initialize project structure, CI/CD, and MCP automation",
                pr_title="feat: Project Scaffolding, CI/CD Pipeline & GitHub MCP Automation",
                pr_body="""## Milestone 1: Scaffolding & GitHub MCP Setup
- Initialized repository structure & configuration files
- Setup GitHub Actions CI/CD pipeline
- Implemented autonomous GitHub MCP workflow automation
- Published comprehensive architecture and benchmark documentation
""",
            )
            print(f"Lifecycle completed: {res.get('branch')}")

        elif args.action == "status":
            print("\nChecking authenticated user status...")
            user_info = client.get_me()
            print(f"User Info: {user_info}")

        elif args.action == "all":
            print("\nExecuting full autonomous lifecycle...")
            report = manager.run_full_autonomous_cycle(project_files=scaffolding_files)
            print("\n" + "=" * 70)
            print(" Full Lifecycle Completed Successfully!")
            print("=" * 70)
            print(f"Repo Setup   : {report.get('repo_setup', {}).get('status', 'OK')}")
            print(f"Issues Added : {len(report.get('issues', []))}")
            print(f"Branch       : {report.get('feature_lifecycle', {}).get('branch')}")
            print(f"Files Pushed : {report.get('feature_lifecycle', {}).get('files_count')}")

        print("\nAll workflow operations executed cleanly.")
        return 0

    except Exception as exc:
        logging.error(f"Workflow execution failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

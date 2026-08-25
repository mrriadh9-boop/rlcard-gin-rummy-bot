# High-Performance RLCard Gin Rummy RL Bot

[![CI / CD Pipeline](https://github.com/mrriadh9-boop/rlcard-gin-rummy-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/mrriadh9-boop/rlcard-gin-rummy-bot/actions)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B%20CUDA-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An autonomous, high-performance reinforcement learning agent engineered to master **Gin Rummy** using **RLCard**, **Maskable Proximal Policy Optimization (PPO)**, **Dual-Stream 2D Convolutional Neural Networks**, **Multi-Agent Self-Play Leagues**, and full **PyTorch CUDA GPU acceleration**.

---

## Key Highlights

- **Custom High-Throughput Vectorized Environment**: Parallel environment wrapper (`VectorGinRummyEnv`) supporting synchronous multi-environment stepping (32–64 parallel games), instant auto-reset, zero-copy PyTorch tensor conversions, and strict 110-action categorical masking.
- **Meld-Aware 2D Convolutional Neural Architecture**: Dual-stream spatial convolutions (`GinRummyNet`) extracting rank patterns (runs) and suit patterns (sets) across the $5 \times 52$ card observation tensor, coupled with a dense feature trunk, masked actor head, and state-value critic head.
- **Maskable PPO with Generalized Advantage Estimation (GAE)**: Numerically stable categorical policy optimization with exact $-\infty$ illegal action logit masking, clipped surrogate objective ($\epsilon=0.2$), entropy regularization, and GAE($\gamma=0.99, \lambda=0.95$).
- **Multi-Agent Self-Play League**: Continuous self-play training pipeline against historical checkpoint pools, curriculum stages, and rule-based baselines to prevent cyclic strategy exploitation.
- **Hierarchical Baselines**: Native `RandomAgent`, RLCard `GinRummyNoviceRuleAgent`, and custom `GinRummyExpertRuleAgent` with meld-aware pickup, speculative honor discards, and intelligent knock timing.
- **Programmatic Tournament & Statistical Suite**: Symmetric seat evaluation (equal games as Player 0 and Player 1) with Wilson 95% score confidence intervals, exact binomial hypothesis testing ($p < 0.01$), and automated metric reporting.
- **Autonomous GitHub MCP CI/CD**: Native integration with the GitHub Model Context Protocol (MCP) server for automated repository provisioning, milestone issue tracking, feature branch syncing, automated peer review, and pull request merging.

---

## System Architecture

```
                                +-----------------------------------+
                                |      VectorGinRummyEnv (CUDA)     |
                                |  (32-64 Parallel Batched Games)   |
                                +-----------------+-----------------+
                                                  |
                        Observation Tensor (B, 5, 52) & Legal Action Mask (B, 110)
                                                  |
                                                  v
                                +-----------------+-----------------+
                                |        GinRummyNet (PyTorch)      |
                                |  - Rank Stream Conv (Runs)        |
                                |  - Suit Stream Conv (Sets)        |
                                |  - Dense Card Trunk (Embedding)   |
                                +--------+-----------------+--------+
                                         |                 |
                          Masked Action Logits        Value V(s)
                                         |                 |
                                         v                 v
                         +---------------+----+    +-------+--------+
                         | CategoricalMasked  |    |  Critic Head   |
                         | Policy Head (110)  |    |  GAE (γ, λ)    |
                         +---------------+----+    +-------+--------+
                                         |                 |
                                         +--------+--------+
                                                  |
                                                  v
                                +-----------------+-----------------+
                                |       Maskable PPO Trainer        |
                                |  - Clipped Surrogate Objective    |
                                |  - Self-Play Opponent League      |
                                +-----------------------------------+
```

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- NVIDIA GPU with CUDA support (e.g. NVIDIA GeForce GTX 1650 or higher)
- PyTorch with CUDA enabled

### Install from Source

```powershell
# Clone the repository
git clone https://github.com/mrriadh9-boop/rlcard-gin-rummy-bot.git
cd rlcard-gin-rummy-bot

# Install dependencies
pip install -r requirements.txt

# Install package in editable development mode
pip install -e .
```

### Verify CUDA Hardware Acceleration

```powershell
python -c "import torch; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Command Line Interface (CLI)

### 1. Train the RL Agent (Vectorized PPO + Self-Play)

```powershell
# Train on GPU with 32 parallel environments
python scripts/train.py --num-envs 32 --total-timesteps 500000 --device cuda --save-dir checkpoints/

# Or using console script entry point
gin-train --num-envs 32 --total-timesteps 500000 --device cuda
```

### 2. Run Head-to-Head Tournament & Benchmarks

```powershell
# Benchmark trained model against Random, Novice, and Expert baselines
python scripts/benchmark.py --model checkpoints/best_model.pt --games 200 --device cuda

# Or using console script entry point
gin-benchmark --model checkpoints/best_model.pt --games 200
```

### 3. Run Autonomous GitHub MCP Workflow

```powershell
# Run full automated release workflow (Repo Check -> Milestone Issues -> Branch -> Push -> PR -> Review -> Merge)
python scripts/run_mcp_workflow.py --owner mrriadh9-boop --repo rlcard-gin-rummy-bot

# Dry-run validation mode (simulates MCP payloads without network mutations)
python scripts/run_mcp_workflow.py --dry-run
```

### 4. Execute Complete Test Suite (Pytest)

```powershell
# Run all E2E test tiers (Tiers 1-4)
pytest -v tests/ --durations=10

# Run with test coverage report
pytest --cov=gin_rummy tests/
```

---

## Reinforcement Learning Formulation

### 1. State Representation ($5 \times 52$)
RLCard partitions the 52 standard playing cards into 5 binary feature planes:
- **Plane 0 (Current Hand)**: Binary indicator of the 10-11 cards currently in the agent's hand.
- **Plane 1 (Top Discard)**: Binary indicator of the card currently on top of the discard pile.
- **Plane 2 (Dead Cards)**: Binary indicator of all discarded/seen cards that are no longer in play.
- **Plane 3 (Opponent Known Cards)**: Cards picked up by the opponent from the discard pile.
- **Plane 4 (Unknown Cards)**: Cards remaining in the stockpile or in opponent's hidden hand.

### 2. Action Space (110 Discrete Actions)
| Action ID Range | Action Type | Description |
|:---:|:---:|---|
| `0` | Score North | Score declaration |
| `1` | Score South | Score declaration |
| `2` | Draw Card | Draw unknown card from stockpile |
| `3` | Pick Up Discard | Take top card from discard pile |
| `4` | Declare Dead Hand | Declare stalemate when stockpile depleted |
| `5` | Gin | Lay off complete hand with 0 deadwood |
| `6 .. 57` | Discard Card `id` | Discard card index `id - 6` |
| `58 .. 109` | Knock Card `id` | Knock and discard card index `id - 58` (deadwood $\le 10$) |

### 3. Maskable Categorical Distribution
Illegal actions are assigned logits of $-\infty$ before computing softmax probabilities:
$$\tilde{z}_a = \begin{cases} z_a & \text{if } a \in \mathcal{A}_{\text{legal}} \\ -\infty & \text{if } a \notin \mathcal{A}_{\text{legal}} \end{cases}$$
$$\pi(a \mid s) = \frac{\exp(\tilde{z}_a)}{\sum_{a' \in \mathcal{A}_{\text{legal}}} \exp(\tilde{z}_{a'})}$$

### 4. Generalized Advantage Estimation (GAE)
Advantages are computed recursively over trajectories:
$$\delta_t^V = r_t + \gamma V(s_{t+1}) (1 - d_t) - V(s_t)$$
$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

---

## Empirical Benchmark Goals & Baselines

| Opponent Agent | Benchmark Match Format | Target Win Rate | Statistical Significance Target |
|---|---|:---:|:---:|
| **RandomAgent** | 200 Symmetric Games (100 P0 / 100 P1) | **$\ge 98.0\%$** | $p < 10^{-15}$ (Binomial Test) |
| **GinRummyNoviceRuleAgent** | 200 Symmetric Games (100 P0 / 100 P1) | **$\ge 65.0\%$** | 95% Wilson CI Lower Bound $> 55.0\%$ |
| **GinRummyExpertRuleAgent** | 200 Symmetric Games (100 P0 / 100 P1) | **$\ge 52.0\%$** | Positive Net Expected Payoff |

---

## Project Repository Layout

```
rlcard_gin_bot/
├── .github/
│   └── workflows/
│       └── ci.yml                      # GitHub Actions Multi-OS CI/CD Pipeline
├── gin_rummy/                          # Core Python Package
│   ├── __init__.py
│   ├── env/
│   │   ├── __init__.py
│   │   ├── vector_env.py               # Parallel batched environment wrapper
│   │   └── custom_scorers.py           # Standard v1 & Zero-Sum scoring functions
│   ├── models/
│   │   ├── __init__.py
│   │   └── neural_net.py               # 2D Conv Rank/Suit + Dense Trunk Network
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── ppo_agent.py                # Maskable PPO Agent with PyTorch CUDA
│   │   ├── expert_rule_agent.py        # Meld-aware heuristic Expert Agent
│   │   └── baseline_wrappers.py        # Uniform wrappers for baseline models
│   ├── training/
│   │   ├── __init__.py
│   │   ├── buffer.py                   # Experience rollout buffer & GAE
│   │   ├── self_play.py                # Multi-agent self-play league manager
│   │   └── trainer.py                  # Main training loop & checkpointing
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark.py                # Symmetric tournament runner
│   │   └── stats.py                    # Wilson CI & Binomial hypothesis testing
│   └── github_workflow/
│       ├── __init__.py
│       └── mcp_automation.py           # Programmatic GitHub MCP Automation Suite
├── tests/                              # E2E Test Suite (Tiers 1-4)
│   ├── __init__.py
│   ├── conftest.py                     # Fixtures & environment generators
│   ├── test_tier1_features.py          # Tier 1: Unit feature coverage
│   ├── test_tier2_boundaries.py        # Tier 2: Boundary & corner cases
│   ├── test_tier3_interactions.py      # Tier 3: Cross-module interactions
│   └── test_tier4_scenarios.py         # Tier 4: Tournament & stress benchmarks
├── scripts/
│   ├── train.py                        # Training CLI entrypoint
│   ├── benchmark.py                    # Evaluation & tournament CLI entrypoint
│   └── run_mcp_workflow.py             # Autonomous GitHub MCP runner
├── requirements.txt                    # Project dependencies
├── pyproject.toml                      # Modern packaging & tooling configuration
├── setup.py                            # Setuptools installation script
├── .gitignore                          # Git ignore rules
└── README.md                           # Master project documentation
```

---

## Autonomous GitHub MCP Workflow Integration

The project includes programmatic automation for managing the full GitHub repository lifecycle via the Model Context Protocol (MCP) server:

1. **Repository Verification / Provisioning**: Ensures `mrriadh9-boop/rlcard-gin-rummy-bot` exists with correct metadata and public visibility.
2. **Milestone Issue Tracking**: Creates structured issues tracking Milestones M1 through M6.
3. **Feature Branch Provisioning**: Automatically branches off `main` to isolate milestone changes.
4. **File Push & Sync**: Atomically pushes code modifications and documentation in structured commits.
5. **Pull Request Lifecycle**: Programmatically opens PRs, submits automated code review approvals (`APPROVE`), and merges via squash commit.

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

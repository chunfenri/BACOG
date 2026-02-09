# BACOG: Budget-Aware Context Generation for Cost-Efficient Multi-Agent Systems

## Overview

BACOG is a **budget-aware** context generation framework for cost-efficient multi-agent workflows. Under a given dollar budget, it follows a pipeline of **Planning + MSIR (minimal-sufficient information retrieval) + Deterministic Compilation + Multi-Worker Execution** to balance **performance** and **cost**. This repository provides evaluation scripts for HotpotQA and QASPER, and supports loading trained policy checkpoints from `checkpoints/`.

## Project Structure

```
BACOG/
├── bacog/                             # Core system implementation (importable as a Python package)
│   ├── configs/                       # Global configuration (budget/pricing/LLM config, etc.)
│   │   └── system_config.py
│   ├── micro_policy/                  # Evidence-set policy (EvidenceSetPolicy) and feature extraction
│   ├── msir/                          # LLM client + planners (MSIRPlanner / DynamicPlanner)
│   ├── data_infrastructure/           # Clues (Clue) and candidate pool management (PoolManager)
│   ├── compiler/                      # Compress/assemble/clip: compile clues into LLM-ready prompts
│   └── execution/                     # Executor, budget scheduler, state management, anytime retry, workers
│       ├── executor.py
│       ├── budget_scheduler.py
│       └── task_state.py
├── scripts/                           # Evaluation entry scripts
│   ├── run_evaluation.py              # HotpotQA (validation) evaluation
│   └── run_evaluation_qasper.py       # QASPER (train/validation/test) evaluation
├── train/                             # Training utilities (policy training / data generation)
│   ├── data_generator.py
│   └── train_policy.py
├── checkpoints/                       # Policy model weights
└── requirements.txt                   # Dependency list
```

## Checkpoints

Pre-trained policy weights (`model_epoch_8.pth`, ~450MB) are too large for GitHub's file size limit. 

**They will be made available on Google Drive / HuggingFace upon acceptance.**

## Key Components

### Core System (`bacog/`)

- **LLM Client (`bacog/msir/llm_client.py`)**: OpenAI SDK v1 compatible wrapper, with JSON mode and usage/cost estimation.
- **Planning**
  - **`DynamicPlanner`**: decomposes a query into sub-queries (Step 0).
  - **`MSIRPlanner`**: selects a minimal set of evidence keys from a budget-feasible candidate window.
- **Micro-Policy (`bacog/micro_policy/model.py`)**
  - **`EvidenceSetPolicy`**: predicts $K$, $B_{in}$, and routing (which worker/intent to use).
- **Execution (`bacog/execution/`)**
  - **`TaskState` (`task_state.py`)**: task state tracking (goal/plan/step/history).
  - **`Executor`**: the main loop that connects “Policy → Budget clipping → MSIR → Compiler → Worker → Anytime retry → accounting”.
  - **`BudgetScheduler`**: derives per-round hard input limits from task-level budget.
- **Workers (`bacog/execution/workers/workers.py`)**
  - Implementations for QA / Search / Summary.

### Configuration System

Configuration is centralized in `bacog/configs/system_config.py` and exposed via `config = SystemConfig()`:

- **Budget/Pricing**
  - `config.budget.P_IN` / `config.budget.P_OUT`: per-token unit prices.
  - `config.budget.B_TASK_TOTAL`: default task budget.
  - `config.msir.AVG_STEP_COST`: average per-step cost used to derive max steps from budget.
- **LLM access config**
  - `config.llm.API_KEY`
  - `config.llm.BASE_URL`

## Experiment Framework

- **Policy loading**: load `EvidenceSetPolicy` weights from `checkpoints/model_epoch_{epoch}.pth`.
- **Evaluation**
  - `scripts/run_evaluation.py`: HotpotQA (`hotpot_qa` `distractor/validation`).
  - `scripts/run_evaluation_qasper.py`: QASPER (`allenai/qasper`).

## Quick Start

### 1) Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure LLM access in `bacog/configs/system_config.py`:

- `config.llm.API_KEY`
- `config.llm.BASE_URL`

### 2) Evaluation (HotpotQA)

```bash
python scripts/run_evaluation.py
```

## Supported Datasets

- **HotpotQA**: `load_dataset("hotpot_qa", "distractor", split="validation")`
- **QASPER**: `load_dataset("allenai/qasper")`

## Notes

- **QASPER `datasets` version constraint**: `scripts/run_evaluation_qasper.py` will error out on `datasets>=4.x`. If you want to run QASPER, install:

```bash
pip install -U "datasets<4"
```

- **Requirements mismatch**: `requirements.txt` currently pins `datasets==4.x`, which is incompatible with the QASPER script. Use separate environments or override the `datasets` version when running QASPER.
- **Model weights**: by default, `checkpoints/model_epoch_8.pth` is used; you can specify via `--epoch`.

## Citation


## License


# SafePilot
### An AI agent that respects authority without stopping legitimate work.

SafePilot is an invoice-processing AI agent designed to make decisions based on user permissions, document trust and approval requirements.

The project explores an important challenge in agent safety: how can an AI agent prevent unauthorized actions while still completing legitimate tasks?

## The Problem

Business documents can contain misleading or malicious instructions. An agent that blindly follows everything it reads could change payment details, bypass approvals or perform actions beyond the user's authority.

SafePilot separates trusted instructions from untrusted document content and uses an independent authorization layer to control sensitive operations.

## Architecture

1. **User request:** Receives the invoice-processing task and user role.
2. **Document inspection:** Reads invoice information and retrieves purchase-order records.
3. **Agent decision:** Selects ACT, ASK_APPROVAL, REFUSE or ASK_CLARIFICATION.
4. **Authorization layer:** Checks identity, role permissions and approval requirements.
5. **Approval workflow:** Requires valid, transaction-bound approval for payment execution.
6. **Audit log:** Records model proposals, policy decisions, approvals and tool execution.

The model proposes actions; the authorization layer enforces permissions.

## AuthorityBench

I created AuthorityBench, a benchmark containing 650 invoice-processing scenarios.

| Split | Scenarios |
|---|---:|
| Training | 400 |
| Validation | 100 |
| Test | 150 |
| **Total** | **650** |

Seven scenario categories:
- Normal workflow
- Approval required
- Indirect prompt injection
- Conflicting instructions
- Unauthorized action
- Ambiguous request
- Approval misuse

## Model Training

Base model: `CohereLabs/tiny-aya-global`

Two LoRA adapters were trained using the same training examples:

- Standard supervised fine-tuning
- Authority-aware supervised fine-tuning

## Evaluation Results

Results on 150 held-out test scenarios:

| Configuration | Decision accuracy | Schema validity |
|---|---:|---:|
| Tiny Aya Global baseline | 14.0% | 0% |
| Standard SFT | 94.7% | 100% |
| Authority-aware SFT | 78.7% | 100% |

### Prompt Ablation

The authority-aware adapter was evaluated with two different system prompts.

| System prompt | Decision accuracy | Schema validity |
|---|---:|---:|
| Standard prompt | 44.0% | 59.3% |
| Authority-aware prompt | 78.7% | 100% |

With the authority-aware prompt, the model correctly classified all approval-misuse and unauthorized-action test cases.

However, it struggled to continue some legitimate workflows.

### Key Finding

Authority-aware behaviour is not simply about refusing unsafe actions. An agent must also recognise when it is safe to continue.

Our experiments show a trade-off between respecting authorization boundaries and completing legitimate tasks. They also demonstrate that system-prompt design has a substantial effect on the behaviour of a fine-tuned model.

## Technology

- Python
- Cohere Tiny Aya Global
- Hugging Face Transformers
- PEFT and LoRA
- PyTorch
- JSONL benchmark and evaluation pipelines
- Pytest

## Project Structure

```text
SafePilot/
├── agent/             # Agent decisions, tools and workflow
├── authorization/    # Identity, policies and enforcement
├── benchmark/        # AuthorityBench dataset and splits
├── data_generation/  # Scenario and trajectory generation
├── environment/      # Invoice, PO and payment environment
├── evaluation/       # Model evaluation and prompt ablation
├── experiments/      # Experimental results
├── tests/            # Automated tests
└── training/         # SFT preparation and training
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/ejaz029/SafePilot.git
cd SafePilot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the test suite:

```bash
pytest -q
```

## Models and Dataset

**Fine-tuned models**

- [Standard SFT Adapter](https://huggingface.co/ejaz5454/safepilot-tiny-aya-standard)
- [Authority-aware SFT Adapter](https://huggingface.co/ejaz5454/safepilot-tiny-aya-authority-aware)

**Dataset**

- [SafePilot AuthorityBench](https://huggingface.co/datasets/ejaz5454/safepilot-authoritybench-sft)

**Model collection**

- [SafePilot – Fine-tuned Models](https://huggingface.co/collections/ejaz5454)

## Limitations and Future Work

The benchmark uses simulated invoice-processing scenarios. The experiments do not establish production-level safety.

Human-in-the-loop evaluation and comparisons across the other Tiny Aya variants remain future work.

Further work will focus on improving legitimate task completion without weakening authorization enforcement.

## About

Developed for the **Lexsi Labs AI Safety & Alignment Hackathon – Track 3: Agentic Fine-tuning**.

GitHub: [ejaz029](https://github.com/ejaz029)

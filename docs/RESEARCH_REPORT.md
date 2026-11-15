# SafePilot — Research Report

## Authority-Aware Agentic Fine-Tuning for Approval-Gated Workflows

**Track:** Track 3 — Agentic Fine-tuning  
**Project:** SafePilot  
**Base Model:** Cohere Tiny Aya Global

## 1. Problem Statement

Tool-using AI agents can interact with business systems and perform multi-step workflows. However, they may follow malicious instructions embedded in untrusted documents, exceed their delegated authority or attempt sensitive actions without the required approval.

SafePilot investigates how fine-tuning can help an agent make better decisions about what it is authorised to do, when it should request approval and when it should refuse.

## 2. Research Question

Can authority-aware fine-tuning reduce unauthorized action proposals while preserving legitimate task completion in tool-using AI agents?

The central challenge is balancing two requirements: preventing unauthorized actions and allowing the agent to complete legitimate work.

## 3. Hypothesis

Training a model on authority-labelled agent trajectories can improve its ability to distinguish trusted instructions from untrusted content and select an appropriate next action:

- **ACT:** Perform an allowed action.
- **ASK_APPROVAL:** Request the required approval.
- **REFUSE:** Reject an unauthorized or unsafe request.
- **ASK_CLARIFICATION:** Request missing information.

This hypothesis is evaluated through supervised fine-tuning, held-out benchmark testing and prompt ablation.

## 4. Application Domain

SafePilot uses a simulated financial invoice-processing workflow.

The agent is expected to:

1. Read an invoice.
2. Verify invoice details against a purchase order.
3. Prepare a legitimate payment.
4. Request approval when required.
5. Execute payment only after valid authorization.
6. Record decisions and tool execution history.

No real financial transactions are performed.

## 5. Research Contributions

### 5.1 AuthorityBench

A benchmark of 650 scenarios covering seven categories:

- Normal workflow
- Approval required
- Indirect prompt injection
- Conflicting instructions
- Unauthorized action
- Ambiguous request
- Approval misuse

| Split | Scenarios |
|---|---:|
| Training | 400 |
| Validation | 100 |
| Held-out test | 150 |
| **Total** | **650** |

Each scenario contains workflow context, user roles, invoice and purchase-order information, trust labels, authorization policies and expected decisions.

### 5.2 Authority-Aware Trajectory Training

A trajectory-generation pipeline produces structured examples for invoice-processing workflows.

The training data includes agent actions and separate human approval events. Payment execution is controlled by the authorization layer rather than learned as an employee agent action.

Two fine-tuning datasets were prepared using the same examples:

- Standard supervised fine-tuning.
- Authority-aware supervised fine-tuning.

### 5.3 Decision-Level Evaluation

The evaluation framework measures model decisions separately from the independent authorization engine.

This distinction is important: an unsafe model proposal does not automatically mean an unauthorized action was executed.

## 6. Experimental Methodology

The experiments use Cohere's Tiny Aya Global model with LoRA fine-tuning.

The main configurations are:

1. Original Tiny Aya Global baseline.
2. Standard supervised fine-tuning.
3. Authority-aware supervised fine-tuning.

The models are evaluated against the same frozen set of 150 held-out scenarios.

A separate prompt ablation compares the authority-aware adapter under two system prompts while keeping the model adapter and test scenarios fixed.

## 7. Experimental Results

### 7.1 Main Comparison

| Configuration | Decision accuracy | Schema validity |
|---|---:|---:|
| Tiny Aya Global baseline | 14.0% | 0% |
| Standard SFT | 94.7% | 100% |
| Authority-aware SFT | 78.7% | 100% |

The standard SFT model achieved the highest overall decision accuracy.

The authority-aware model correctly classified all approval-required, approval-misuse and unauthorized-action test cases. However, it struggled with some legitimate workflows and conflicting instructions.

### 7.2 Prompt Ablation

The authority-aware adapter was tested with two system prompts.

| System prompt | Decision accuracy | Schema validity |
|---|---:|---:|
| Standard prompt | 44.0% | 59.3% |
| Authority-aware prompt | 78.7% | 100% |

The authority-aware prompt improved decision accuracy by 34.7 percentage points and schema validity by 40.7 percentage points.

This indicates that the instructions used during inference have a substantial effect on the behaviour of the fine-tuned adapter.

### 7.3 Category-Level Observations

With the authority-aware prompt, the model achieved:

- Approval required: 22/22 correct.
- Approval misuse: 21/21 correct.
- Unauthorized action: 21/21 correct.
- Ambiguous request: 21/21 correct.
- Normal workflow: 11/22 correct.
- Conflicting instructions: 7/21 correct.
- Indirect prompt injection: 15/22 correct.

The main weakness was unnecessary clarification when the agent should have continued with a safe action, such as reading or verifying information.

## 8. Key Findings

The experiments reveal three important observations.

**First, standard fine-tuning improved overall task decision accuracy.** The standard SFT model achieved 94.7% accuracy compared with 14% for the baseline.

**Second, authority-aware behaviour improved authorization-related decisions.** With the authority-aware prompt, the model correctly handled every approval-misuse and unauthorized-action test case.

**Third, authority-aware behaviour introduced a task-completion trade-off.** The model sometimes stopped legitimate workflows instead of continuing with safe read-only actions.

The prompt ablation further showed that the authority-aware adapter performs substantially better when paired with an authority-aware system prompt.

## 9. Safety Architecture

The model is not the security boundary.

An independent authorization engine validates:

- User identity and role permissions.
- Whether an action is allowed.
- Payment and transaction details.
- Approval validity and expiry.
- Transaction binding and one-time approval consumption.

The payment simulator must reject unauthorized execution even if the model proposes an unsafe action.

Audit records distinguish model proposals, policy decisions, human approvals and tool executions.

## 10. Evaluation Scope and Limitations

The current evaluation establishes decision accuracy and schema validity on a simulated held-out benchmark.

The following limitations remain:

- The benchmark represents one simulated invoice-processing domain.
- The main comparison changes both the adapter and, for the authority-aware model, the system prompt. It is not a pure adapter-only comparison.
- Prompt ablation isolates the system prompt effect for the authority-aware adapter.
- Decision accuracy and schema validity do not independently establish the rate of unauthorized tool execution.
- Human-in-the-loop evaluation has not yet been completed.
- Comparisons across the other three Tiny Aya variants remain future work.
- The results do not establish production-level safety.

## 11. Future Work

Further research should focus on:

1. Improving legitimate workflow completion without weakening authorization boundaries.
2. Evaluating human approval responses and simulated LLM-based approval responses separately.
3. Comparing the other Tiny Aya variants.
4. Measuring raw unauthorized action proposals separately from actual tool execution.
5. Expanding the benchmark to additional business workflows.

## 12. Conclusion

SafePilot investigates whether authority-aware fine-tuning can help tool-using AI agents respect permissions and approval requirements.

The results show that standard fine-tuning can improve overall task performance, while authority-aware instructions can improve authorization-related decisions. However, the model must also learn when it is safe to continue a legitimate workflow.

The central finding is that a safe agent should not simply refuse more often. It should understand the boundary between actions it can perform, actions requiring approval and actions it must not perform.

**Project:** SafePilot  
**GitHub:** https://github.com/ejaz029/SafePilot
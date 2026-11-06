# SafePilot — Research Definition

## Project Title

SafePilot: Authority-Aware Agentic Fine-Tuning for Approval-Gated Workflows

## Track

Track 3 — Agentic Fine-tuning

## Problem Statement

Tool-using AI agents can interact with business systems and execute
multi-step workflows. However, they may follow malicious instructions
embedded in untrusted documents, exceed their delegated authority, or
attempt sensitive actions without required approval.

## Research Question

Can authority-aware fine-tuning reduce unauthorized action proposals
while preserving legitimate task completion in tool-using AI agents?

## Hypothesis

Training a model on authority-labelled agent trajectories will improve
its ability to distinguish trusted instructions from untrusted content
and select appropriate actions:

- ACT
- ASK_APPROVAL
- REFUSE
- ASK_CLARIFICATION

This hypothesis will be evaluated experimentally.

## Application Domain

Simulated financial invoice processing.

The agent will:

1. Read an invoice.
2. Verify invoice details against a purchase order.
3. Prepare a legitimate payment.
4. Request approval when required.
5. Execute payment only after valid authorization.
6. Record decisions and tool execution history.

No real financial transactions will be performed.

## Research Contributions

### 1. AuthorityBench

A benchmark of business workflow scenarios containing trusted
instructions, untrusted document content, authorization boundaries,
approval requirements, conflicting instructions and adversarial cases.

### 2. Authority-Aware Trajectory Training

A fine-tuning dataset designed to teach an agent when to act,
request approval, refuse or ask for clarification.

### 3. Decision-Level Evaluation

An evaluation framework measuring model decisions separately from
the independent authorization engine.

## Experimental Conditions

A. Original Cohere Tiny Aya Global model.

B. Standard tool-use supervised fine-tuning.

C. SafePilot authority-aware supervised fine-tuning.

All conditions will use the same simulated environment, tools,
authorization policy and held-out evaluation scenarios.

## Primary Evaluation Metrics

- Decision accuracy
- Unauthorized action proposal rate
- Indirect prompt-injection success rate
- Legitimate task completion rate
- False refusal rate
- Approval precision and recall
- Unauthorized action execution rate
- Invalid approval acceptance rate

Model proposals and actual tool executions will be measured separately.

## Safety Architecture

The model is not the security boundary.

An independent authorization engine will validate permissions,
transaction details and approval tokens before sensitive tool execution.

The payment simulator must reject unauthorized execution even if
the model proposes an unsafe action.

## Technology

- Cohere Tiny Aya Global
- Hugging Face Transformers
- TRL
- PEFT / LoRA
- AgentTune
- AuditKIT
- Python
- Streamlit
- Git and GitHub
- Hugging Face Hub

## Scope and Limitations

The initial implementation focuses on one simulated invoice workflow.

It does not connect to real banking systems or execute real payments.

Research novelty and performance improvements must be established
through comparison and evaluation, not assumed in advance.

## Current Research Status

Research direction defined.
Implementation and experimental validation pending.

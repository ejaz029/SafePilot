# SafePilot

**Authority-Aware Agentic Fine-Tuning for Approval-Gated Workflows**

SafePilot investigates whether authority-aware fine-tuning can improve
the decisions of tool-using AI agents in business workflows.

## Research Question

Can authority-aware fine-tuning reduce unauthorized action proposals
while preserving legitimate task completion?

## Problem

AI agents can interact with business systems, but may follow malicious
instructions embedded in untrusted documents or attempt actions beyond
their delegated authority.

## Proposed Approach

SafePilot uses a simulated invoice-processing workflow to investigate
authority-aware agent behaviour.

The agent must distinguish between:

- Authorized actions
- Actions requiring human approval
- Unauthorized actions
- Ambiguous requests requiring clarification

## Experimental Comparison

Three model conditions will be evaluated:

1. Original Cohere Tiny Aya Global
2. Standard tool-use supervised fine-tuning
3. SafePilot authority-aware supervised fine-tuning

## Architecture

- Cohere model for agent decisions
- AgentTune for agent execution and trajectory collection
- Independent authorization engine
- Simulated invoice and payment tools
- AuthorityBench evaluation dataset
- AuditKIT and custom evaluation metrics

## Safety

All payment operations are simulated.

The authorization engine independently enforces permissions and
approval requirements. Model decisions and actual tool executions
are recorded separately.

## Project Status

Research definition and initial project foundation established.
Implementation and experimental validation are in progress.

## Repository Structure

- `agent/` — Agent orchestration
- `authorization/` — Permission and approval enforcement
- `benchmark/` — Evaluation scenarios
- `configs/` — Project configuration
- `data_generation/` — Dataset generation
- `demo/` — Interactive demonstration
- `docs/` — Research and architecture documentation
- `environment/` — Business workflow simulator
- `evaluation/` — Metrics and experiments
- `training/` — Fine-tuning pipelines
- `tests/` — Automated tests

## Getting Started

Detailed setup and execution instructions will be added as the
implementation becomes available.

## Research Integrity

Performance improvements will be reported only after controlled
experiments on held-out scenarios. No experimental results are
claimed at this stage.

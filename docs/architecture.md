# SafePilot — System Architecture

## 1. Overview

SafePilot is an authority-aware agent operating inside a simulated
financial invoice-processing workflow.

The system separates model decision-making from security enforcement.

## 2. Components

### 2.1 Input Layer

Receives:
- User instructions
- Invoice documents
- Purchase order records
- Approval context

Document content is treated as untrusted data.

### 2.2 Agent Layer

The Cohere model selects one structured decision:

- ACT
- ASK_APPROVAL
- REFUSE
- ASK_CLARIFICATION

The agent proposes tool calls with validated arguments.

### 2.3 Authorization Layer

Independently checks:
- User identity
- Role permissions
- Requested operation
- Transaction amount
- Beneficiary
- Approval validity
- Approval-to-transaction binding

The model cannot bypass this layer.

### 2.4 Tool Execution Layer

Initial tools:

1. read_invoice
2. lookup_purchase_order
3. prepare_payment
4. request_approval
5. execute_payment

All payment operations occur in a simulated environment.

### 2.5 Audit Layer

Records:
- Scenario identifier
- Model condition and revision
- Model decision
- Proposed tool and arguments
- Authorization result
- Actual tool execution
- Approval details
- Final workflow state

Model proposals and actual executions must remain distinguishable.

### 2.6 Evaluation Layer

Runs benchmark scenarios against three model conditions:

- Original model
- Standard SFT
- Authority-aware SFT

Computes decision, safety and task-completion metrics.

## 3. Main Workflow

1. Receive a user request.
2. Read the invoice.
3. Retrieve the corresponding purchase order.
4. Compare invoice and purchase order details.
5. Prepare a payment when the records are valid.
6. Determine whether approval is required.
7. Request approval from an authorized approver.
8. Validate the approval token.
9. Execute the simulated payment.
10. Record the final state and audit history.

## 4. Security Invariants

- Invoice content cannot grant permissions.
- Tool arguments must pass schema validation.
- The agent cannot approve its own restricted transaction.
- Approval must be bound to the exact transaction.
- Expired or reused approval tokens must be rejected.
- Unauthorized payment execution must be blocked.
- All sensitive execution attempts must be recorded.

## 5. Integration Boundaries

AgentTune:
- Agent execution and trajectory collection.
- Training workflow integration where supported.

AuditKIT:
- Agent evaluation and adversarial testing where supported.

SafePilot:
- Business simulator.
- Authorization engine.
- AuthorityBench.
- Research-specific metrics.
- Demo application.

## 6. Execution Principle

The model proposes.
The authorization engine decides whether execution is permitted.
The simulator executes only permitted actions.
The audit system records what actually happened.
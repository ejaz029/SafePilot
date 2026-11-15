# SafePilot

## Authority Aware Agentic Fine-Tuning for Approval Gated Workflows

**Research Area:** AI Agent Safety, Alignment and Fine-Tuning
**Base Model:** Cohere Tiny Aya Global
**Author:** Ejaz Belgaum

---

## Abstract

AI agents are becoming capable of interacting with external tools, accessing business information and completing multi-step workflows. However, giving an agent access to tools also introduces a difficult problem: it must understand not only what it can do, but also what it is authorised to do.

An agent processing an invoice may encounter malicious instructions embedded in the document, conflicting requests from a user or an attempt to bypass an approval requirement. Simply following the latest instruction can result in actions that violate the intended permissions of the workflow. On the other hand, making an agent excessively cautious can prevent it from completing legitimate tasks.

SafePilot investigates whether authority-aware fine-tuning can help tool-using AI agents distinguish trusted instructions from untrusted document content, respect delegated permissions and preserve legitimate task completion.

The project introduces AuthorityBench, a benchmark containing 650 simulated invoice-processing scenarios. Two LoRA adapters were trained using Cohere's Tiny Aya Global model: one using standard supervised fine-tuning and another using authority-aware instructions. Both were trained on the same examples and evaluated against 150 held-out scenarios.

The standard SFT model achieved 94.7% decision accuracy, compared with 14% for the original baseline. The authority-aware model achieved 78.7% decision accuracy and correctly handled all test cases involving approval misuse and unauthorized actions when paired with its authority-aware system prompt.

A prompt ablation further revealed that the authority-aware adapter's decision accuracy changed from 44% with a standard prompt to 78.7% with an authority-aware prompt.

The results reveal an important trade-off: an agent can become better at respecting authorization boundaries while becoming less effective at continuing legitimate workflows. Safe agent behaviour therefore requires more than refusing unsafe actions. It also requires understanding when an action is safe to perform.

---

# 1. Introduction

AI agents are moving beyond question answering. They can read documents, retrieve records, call APIs and perform sequences of actions through external tools.

This creates a different category of risk from ordinary language generation.

A language model that generates an incorrect explanation may provide misleading information. An agent connected to business tools may turn an incorrect decision into an attempted operation.

Consider a simple invoice-processing workflow.

An employee asks an agent to verify an invoice against a purchase order and prepare a payment. The invoice contains an additional note instructing the agent to change the beneficiary and release the payment immediately.

The agent must recognise several things:

- The employee's request is trusted only within the employee's permissions.
- The invoice is information to inspect, not an authority that can issue commands.
- Purchase-order information and authorization policies define the permitted workflow.
- Payment execution requires the appropriate approval and authorization.
- A malicious instruction inside a document must not override these boundaries.

The challenge is not simply to make the agent refuse suspicious requests.

The agent should still be able to read the invoice, retrieve the purchase order, identify discrepancies and prepare legitimate work. It should stop only when the next action requires clarification, approval or refusal.

SafePilot explores this distinction through a simulated business workflow and supervised fine-tuning experiments.

# 2. Research Question and Hypothesis

## 2.1 Research question

**Can authority-aware fine-tuning reduce unauthorized action proposals while preserving legitimate task completion in tool-using AI agents?**

This question examines two related objectives:

1. Preventing the agent from selecting actions that exceed its authority.
2. Preserving its ability to complete legitimate tasks without unnecessary refusal or clarification.

These objectives can conflict. A model that becomes overly cautious may avoid unsafe actions but also fail to perform actions that are clearly permitted.

## 2.2 Hypothesis

Training a model on authority-labelled agent trajectories can improve its ability to distinguish trusted instructions from untrusted content and select an appropriate next action.

SafePilot uses four decision categories:

| Decision          | Meaning                                  |
| ----------------- | ---------------------------------------- |
| ACT               | Perform an allowed action                |
| ASK_APPROVAL      | Request required approval                |
| REFUSE            | Reject an unauthorized or unsafe request |
| ASK_CLARIFICATION | Request missing information              |

The hypothesis is evaluated experimentally rather than assumed to be true.

# 3. Application Domain: Invoice Processing

SafePilot uses a simulated financial invoice-processing environment.

The agent is expected to follow a sequence of operations:

1. Read an invoice.
2. Retrieve the corresponding purchase order.
3. Verify invoice details against the purchase order.
4. Prepare a legitimate payment.
5. Request approval when required.
6. Execute payment only after valid authorization.
7. Record decisions and tool execution history.

No real financial transactions are performed.

## 3.1 Trust boundaries

The workflow distinguishes between different sources of information.

| Source                 | Trust treatment                        |
| ---------------------- | -------------------------------------- |
| User request           | Trusted within actual role permissions |
| Invoice content        | Untrusted document data                |
| Purchase-order records | Trusted workflow data                  |
| Authorization policy   | Trusted system policy                  |
| Approval record        | Validated authorization evidence       |

This distinction is important because content can contain instructions without having the authority to issue them.

For example, an invoice may contain text requesting an immediate beneficiary change. The agent can identify and report that text, but it must not treat it as an authorized command.

# 4. SafePilot Architecture

A central design decision in SafePilot is separating model decision-making from authorization enforcement.

The model is responsible for selecting a proposed next action. An independent authorization layer determines whether that action is permitted.

## 4.1 Architecture components

**Agent layer**

The agent receives the user request and workflow context, then selects one of the four decision categories.

**Tool layer**

The agent interacts with the simulated invoice and payment environment through defined tools, including:

- read_invoice
- lookup_purchase_order
- prepare_payment
- request_approval
- approve_payment
- execute_payment

**Business environment**

The environment maintains invoice records, purchase orders, payment records and approval information.

**Authorization layer**

The authorization system checks user identity, role permissions and action-specific requirements.

**Approval manager**

Approval tokens are checked for validity, expiry, transaction binding and one-time consumption.

**Audit layer**

The audit system records model proposals, policy decisions, human approvals and tool executions as separate events.

## 4.2 Why authorization is independent of the model

The model is not the security boundary.

Even if the model produces an unsafe tool proposal, the authorization layer must reject execution when the required permissions or approval are missing.

For payment execution, the system requires a valid approval bound to the transaction and Finance Admin authorization.

This separation reduces dependence on the model making the correct decision every time.

It also makes evaluation more meaningful: a model proposal and an actual executed action are different events.

# 5. AuthorityBench: Benchmark Construction

To evaluate authority-aware behaviour, I created AuthorityBench.

The benchmark contains 650 scenarios designed around invoice-processing workflows, authorization boundaries and instruction conflicts.

## 5.1 Dataset composition

| Split      | Scenarios | Purpose             |
| ---------- | --------: | ------------------- |
| Training   |       400 | Fine-tuning         |
| Validation |       100 | Training evaluation |
| Test       |       150 | Held-out evaluation |
| **Total**  |   **650** |                     |

The held-out test set is frozen and used for model comparisons.

## 5.2 Scenario categories

AuthorityBench covers seven categories.

**Normal workflow**

The user requests legitimate invoice-processing operations within their permissions.

**Approval required**

The workflow reaches an operation that requires approval.

**Indirect prompt injection**

An invoice or other untrusted document contains instructions intended to influence the agent.

**Conflicting instructions**

Instructions in document content conflict with trusted workflow requirements.

**Unauthorized action**

The requested action exceeds the user's actual permissions.

**Ambiguous request**

The agent lacks sufficient information to choose a safe next action.

**Approval misuse**

The scenario attempts to misuse, bypass or improperly claim approval authority.

## 5.3 Scenario structure

Each scenario contains structured information such as:

- Scenario identifier and category.
- Workflow phase.
- User request.
- Invoice content.
- Purchase-order information.
- User role.
- Trust labels.
- Authorization policy.
- Workflow state.
- Expected decision.
- Expected tool and arguments.
- Expected tool sequence.
- Expected final state.

The benchmark evaluates the next-action decision as well as workflow expectations.

The distinction matters because selecting the correct next action does not automatically prove that the entire workflow was completed correctly.

# 6. Agent Trajectory Generation

A separate trajectory-generation pipeline was developed to prepare agent interaction examples for supervised fine-tuning.

The pipeline uses the simulated business environment to generate structured workflow records.

The completed generation run processed 400 training scenarios:

| Generation result            | Count |
| ---------------------------- | ----: |
| Input scenarios              |   400 |
| Successful scenarios         |   172 |
| Skipped scenarios            |   228 |
| Failed scenarios             |     0 |
| Generated trajectory records | 1,032 |

The 172 successful workflows each produced six records.

These records included agent actions and a separate human approval event.

The recorded event distribution was:

| Event type            |   Records |
| --------------------- | --------: |
| Agent actions         |       860 |
| Human approval events |       172 |
| **Total**             | **1,032** |

The trajectory pipeline keeps approval events separate from agent actions. Payment execution is controlled by the authorization layer and is not treated as an employee agent action.

The resulting records provide structured examples of how the agent progresses through a workflow.

# 7. Fine-Tuning Methodology

SafePilot uses Cohere's Tiny Aya Global model as its base model.

Two LoRA adapters were trained.

## 7.1 Standard supervised fine-tuning

The standard adapter was trained using the prepared task examples and standard workflow instructions.

It provides a comparison point for studying the effect of introducing authority-specific instructions.

## 7.2 Authority-aware supervised fine-tuning

The authority-aware adapter used the same training examples and labels, with system instructions that explicitly describe:

- User role permissions.
- The untrusted nature of invoice content.
- The trusted status of purchase-order and authorization information.
- Approval requirements.
- Restrictions on payment execution.
- Conditions for clarification and refusal.

Using the same examples helps isolate the influence of authority-aware instructions during training, although the later main inference comparison also involves different system prompts.

## 7.3 Training configuration

| Parameter           | Value                           |
| ------------------- | ------------------------------- |
| Base model          | CohereLabs/tiny-aya-global      |
| Fine-tuning method  | LoRA                            |
| Training examples   | 916                             |
| Validation examples | 100                             |
| Epochs              | 2                               |
| Optimizer steps     | 230                             |
| Training hardware   | T4 GPU                          |
| Adapter type        | Parameter-efficient fine-tuning |

The two adapters were trained separately.

The authority-aware adapter used approximately 0.4492% trainable parameters.

Both training runs completed and the resulting adapters were uploaded to Hugging Face.

# 8. Experimental Design

The main evaluation examined three configurations:

**Experiment A — Original baseline**

The original Tiny Aya Global model without the SafePilot fine-tuned adapters.

**Experiment B — Standard SFT**

The standard fine-tuned adapter with the standard system prompt.

**Experiment C — Authority-aware SFT**

The authority-aware adapter with the authority-aware system prompt.

All three configurations were evaluated against the same 150 held-out AuthorityBench scenarios.

A separate prompt ablation was then conducted using the authority-aware adapter with two different system prompts.

This additional experiment helps examine whether the observed behaviour is influenced by the adapter, the system prompt or their interaction.

# 9. Main Experimental Results

## 9.1 Decision accuracy and schema validity

The main comparison produced the following results:

| Configuration            | Correct decisions | Decision accuracy | Schema validity |
| ------------------------ | ----------------: | ----------------: | --------------: |
| Tiny Aya Global baseline |            21/150 |             14.0% |              0% |
| Standard SFT             |           142/150 |             94.7% |            100% |
| Authority-aware SFT      |           118/150 |             78.7% |            100% |

The standard SFT configuration achieved the highest overall decision accuracy.

Both fine-tuned configurations achieved perfect schema validity in the main comparison.

The baseline struggled to follow the required structured decision format.

## 9.2 Category-level results

The authority-aware model produced the following results with its authority-aware prompt:

| Category                  | Correct | Total | Accuracy |
| ------------------------- | ------: | ----: | -------: |
| Approval required         |      22 |    22 |     100% |
| Approval misuse           |      21 |    21 |     100% |
| Unauthorized action       |      21 |    21 |     100% |
| Ambiguous request         |      21 |    21 |     100% |
| Indirect prompt injection |      15 |    22 |    68.2% |
| Normal workflow           |      11 |    22 |      50% |
| Conflicting instructions  |       7 |    21 |    33.3% |

The model handled approval-related and unauthorized-action scenarios particularly well.

However, performance was weaker on conflicting instructions and normal workflows.

This suggests that the model sometimes has difficulty distinguishing an unsafe requested operation from a safe intermediate action that should still be performed.

## 9.3 Decision errors

A review of incorrect authority-aware predictions showed a recurring pattern.

Several scenarios expected the agent to perform a safe read-only action, such as looking up a purchase order.

Instead, the model selected ASK_CLARIFICATION.

In these cases, the malicious or conflicting content did not necessarily make the next read-only action unsafe.

The model appeared to stop the workflow too early.

This is an important distinction: preventing an unsafe action should not require abandoning every safe action that comes before it.

# 10. Prompt Ablation

To investigate the effect of inference instructions, I evaluated the same authority-aware adapter using two system prompts.

The model adapter and held-out scenarios remained unchanged.

Only the system prompt was varied.

| Prompt condition       | Correct decisions | Decision accuracy | Schema validity |
| ---------------------- | ----------------: | ----------------: | --------------: |
| Standard prompt        |            66/150 |             44.0% |           59.3% |
| Authority-aware prompt |           118/150 |             78.7% |            100% |

The authority-aware prompt improved decision accuracy by 34.7 percentage points.

Schema validity increased by 40.7 percentage points.

## 10.1 Category-level comparison

| Category                  | Standard prompt | Authority-aware prompt |
| ------------------------- | --------------: | ---------------------: |
| Approval required         |            100% |                   100% |
| Approval misuse           |              0% |                   100% |
| Unauthorized action       |              0% |                   100% |
| Ambiguous request         |            100% |                   100% |
| Indirect prompt injection |           81.8% |                  68.2% |
| Normal workflow           |           22.7% |                    50% |
| Conflicting instructions  |              0% |                  33.3% |

The authority-aware prompt substantially improved authorization-related decisions.

However, indirect prompt injection accuracy was lower in this particular test, illustrating that improvements are not uniform across every category.

## 10.2 Interpretation

The results demonstrate a strong interaction between the fine-tuned adapter and the system prompt.

A model trained to follow authority-aware instructions may not behave consistently when those instructions are removed or replaced.

This is important when evaluating fine-tuned agents: adapter performance cannot always be understood independently of the inference prompt.

The ablation also suggests that model training and inference instructions should be designed and evaluated together.

# 11. What the Experiments Reveal

The experiments reveal three main observations.

### 11.1 Standard fine-tuning improves overall task decisions

Standard SFT achieved 94.7% decision accuracy compared with 14% for the original baseline.

This indicates that supervised examples helped the model follow the structured workflow decision format and select the expected next actions.

### 11.2 Authority-aware instructions improve authorization-related decisions

The authority-aware configuration correctly classified all approval-misuse and unauthorized-action scenarios in the held-out test set.

This is promising evidence that authority-related instructions can influence agent decision-making.

However, these results should not be interpreted as proof that unauthorized tool execution has been eliminated.

### 11.3 Excessive caution can reduce usefulness

The authority-aware model frequently selected clarification when a safe intermediate action was expected.

This created a trade-off between respecting authorization boundaries and continuing legitimate work.

A useful agent must understand that it can safely inspect information or verify records without automatically accepting malicious instructions contained in those records.

The desired behaviour is not simply to refuse more often.

It is to select the safest valid next action.

# 12. Authorization Enforcement and System Safety

SafePilot does not rely exclusively on model behaviour for security.

The independent authorization layer checks whether the proposed action can be executed.

Its responsibilities include:

- Identity and role validation.
- Permission checks.
- Payment preparation restrictions.
- Approval validation.
- Transaction binding.
- Approval expiry.
- One-time approval consumption.
- Payment execution authorization.

The audit system records separate events for model proposals, policy decisions, human approvals and tool executions.

This makes it possible to distinguish:

1. What the model proposed.
2. What the authorization layer permitted or rejected.
3. Whether a tool was actually executed.

This distinction is essential when studying agent safety.

A model may propose an unauthorized action, but the enforcement layer should reject it. Conversely, a model's correct refusal decision does not by itself demonstrate that the entire application is secure.

# 13. Limitations

The current work has several limitations.

**Simulated application domain**

The benchmark focuses on invoice-processing workflows. Its results should not be assumed to generalise to every business application.

**Limited model comparison**

The completed experiments use Tiny Aya Global. Comparisons with the other three Tiny Aya variants remain future work.

**Human approval evaluation**

The trajectory generator includes separate human approval events, but a dedicated human-in-the-loop evaluation has not yet been completed.

**Decision accuracy is not execution safety**

The main reported metrics measure model decision accuracy and output schema validity. They do not independently establish the rate of unauthorized action proposals or actual unauthorized execution.

**Prompt and adapter interaction**

The main comparison uses different system prompts for the standard and authority-aware configurations. It is therefore not a pure adapter-only comparison.

**Task completion trade-off**

The authority-aware configuration still struggles with some legitimate workflows, particularly conflicting-instruction scenarios.

These limitations define what the current experiments establish and what requires further investigation.

# 14. Future Research

The next stage of SafePilot should focus on improving the balance between safety and useful task completion.

Potential directions include:

1. Training on examples where malicious document content is present but safe read-only actions should continue.
2. Improving the distinction between refusing an unsafe operation and asking for clarification about missing information.
3. Evaluating human approval responses separately from simulated LLM approval responses.
4. Comparing additional Tiny Aya variants.
5. Measuring raw unauthorized action proposals independently from tool execution.
6. Expanding the benchmark to other approval-gated business workflows.
7. Evaluating the consistency of model behaviour across different system prompts.

An especially important direction is teaching the agent to continue safe parts of a workflow even when it encounters untrusted or conflicting content.

# 15. Conclusion

SafePilot investigates whether authority-aware fine-tuning can improve decision-making in AI agents operating through approval-gated workflows.

The experiments show that standard supervised fine-tuning can substantially improve overall decision accuracy. Authority-aware fine-tuning, when paired with an appropriate system prompt, performs strongly on approval-misuse and unauthorized-action scenarios.

However, the results also reveal a significant limitation: an agent can become overly cautious and fail to complete legitimate work.

The central lesson is that agent safety is not simply about preventing actions. It is about selecting the correct action within the available authority.

A capable agent should know when to act, when to request approval, when to clarify and when to refuse.

SafePilot provides an initial experimental framework for studying that balance.

---

# Reproducibility and Resources

The project source code, benchmark, adapters and research artifacts are available through the following repositories.

**Source code**

[https://github.com/ejaz029/SafePilot](https://github.com/ejaz029/SafePilot)

**Fine-tuned models**

- Standard SFT: [https://huggingface.co/ejaz5454/safepilot-tiny-aya-standard](https://huggingface.co/ejaz5454/safepilot-tiny-aya-standard)
- Authority-aware SFT: [https://huggingface.co/ejaz5454/safepilot-tiny-aya-authority-aware](https://huggingface.co/ejaz5454/safepilot-tiny-aya-authority-aware)

**Benchmark and training data**

[https://huggingface.co/datasets/ejaz5454/safepilot-authoritybench-sft](https://huggingface.co/datasets/ejaz5454/safepilot-authoritybench-sft)

**Model collection**

[https://huggingface.co/collections/ejaz5454](https://huggingface.co/collections/ejaz5454)

**Research report**

[https://github.com/ejaz029/SafePilot/blob/main/docs/RESEARCH_REPORT.md](https://github.com/ejaz029/SafePilot/blob/main/docs/RESEARCH_REPORT.md)

**Project status:** Experimental implementation and initial evaluation completed. Further human-in-the-loop and cross-model evaluation remain open research directions.

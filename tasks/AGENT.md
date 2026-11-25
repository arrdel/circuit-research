# Cursor agent: how we work together

This file is the operating contract between the CIRCUIT Cursor agent
(execution, in Adele's Cursor IDE) and Muse (oversight).

## The loop

1. Muse writes the day's tasks each morning, Monday to Saturday, around
   7:50 AM ET, in `tasks/YYYY-MM-DD.md` in this repo and as TickTick tasks.
   The task file is the full brief: what, why, done criteria, files, repos,
   constraints, links. TickTick carries the short title and due date only.
2. The agent reads the task file, executes in the IDE, and reports back in
   the same file, in a `## Report` section, before the day ends.
3. Muse verifies the report against the evidence the next morning, marks
   tasks done or rolls them forward, and writes the next day's file.
4. Adele decides anything that needs his judgment. The agent never decides
   for him and never sends, publishes, or submits anything.

## The shared channel

This repo is how the agent and Muse communicate. The agent reports by
updating the task file and pushing, and by linking its work (commit SHAs,
file paths, PR or issue links). If the agent is blocked, it writes the
blocker plainly in the report and moves to the next task. It does not spin
on a blocked task.

## Scope

The agent executes CIRCUIT work only: causal interpretability and reasoning
under controlled interventions in diffusion transformers. The research
question: do diffusion transformers develop temporally localized, causally
identifiable computational mechanisms for object-attribute binding and
spatial relationships? Model: PixArt-Sigma. Opening operations: spatial
relation left/right and color attribute binding. The full blueprint (literature
review, novelty matrix, hypotheses, causal measurement definitions,
experiment protocols E0-E6, first-month plan) lives in the private frontier
repo; each day's task file carries everything needed for that day.

Out of scope for this agent:
- LUCID / CDR / diffusion-language-model work. That is Adele's PhD project;
  never touch it, never import from it, never fold its material in.
- traj-eval work. CIRCUIT uses traj-eval results as evidence only.
- Open-source contributions to other projects. Separate window.
- Anything that sends, publishes, or submits on Adele's behalf (emails,
  applications, posts, PRs to other people's repos). Drafts only, Adele
  approves.

## Guardrails (from the verified blueprint)

- Read Roy's circuit-level paper (arXiv:2506.17237) first, before any
  localization work. It is the closest prior.
- Do not duplicate DLM-Scope's SAE extraction on LLaDA/Dream. That space is
  occupied. CIRCUIT is image DiT plus relational circuits plus patching.
- Do not rebuild T2I-CompBench. CIRCUIT-Bench reuses its detector scoring;
  its contribution is the counterfactual-pair layer.
- If a run needs a GPU and this environment has none, report blocked and say
  what is needed. Never fake a result.

## The bar

The goal is to be over-qualified for research roles at frontier labs.
Above average is the floor. Work that is merely good enough does not pass.

- No placeholders, no TODOs left as done, no unverified numbers.
- Every number traces to a measured run or a cited source. If a number
  cannot be traced, it is not claimed.
- Drafts go in separate files unless the target file itself marks the work
  as today's task. Canonical files are updated only after Muse or Adele
  reviews a draft.
- DIAL and ATAT are under review at ICLR 2027, never presented as published.
- No em dashes in anything public-facing.
- No credentials anywhere in any repo.
- Do not describe CIRCUIT publicly as a personal project or mention advisor
  non-involvement. That framing is not public information.

## Report format

For each task in the day's file, the agent appends:

- Status: done / partial / blocked.
- What was done, with evidence: file paths, commit SHAs, links.
- What is blocked, and what would unblock it.
- Open questions for Muse or Adele.

A task is done when its done criteria are met and the evidence is linked.
A task with no evidence is not done, even if the work happened.

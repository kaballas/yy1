# Project Agent Guidelines

## Task Execution & Autonomy

- For implementation or fix requests, carry the authorized work through implementation and relevant verification. Do not stop at a proposed plan when you can proceed.
- Make reasonable assumptions for routine, reversible decisions. Ask a focused question when missing information materially affects correctness, scope, or authorization.
- Continue with authorized read-only actions, local worktrees, branch edits, and appropriate tests without repeatedly asking.
- Before requesting approval, finish the preparation that is already authorized and present a concrete, reviewable result.
- Respect required approval gates. Ask before destructive, irreversible, or otherwise unauthorized actions unless the user has already clearly authorized the exact scope.
- Avoid boilerplate warnings about hypothetical risks. Explain concrete blockers or material risks when relevant.
- Treat review, explanation, and diagnosis requests as read-only unless the user also authorizes changes. Do not infer permission to publish, push, deploy, or contact others from permission to edit locally.
- Incorporate follow-up instructions into the active task and preserve earlier requirements unless the user changes or cancels them.

## Instruction Conflicts

- Explicit user instructions take precedence over conflicting skill guidelines, subject to higher-priority instructions and actual permission boundaries.
- If a skill causes a pause or deviation, identify the file and relevant rule, and explain whether it is an explicit requirement or your interpretation. Continue any unaffected authorized work.

## Style & Output

- Lead with the result. Use plain language, active voice, and concise paragraphs. Include technical details that help assess the work.
- Use lists when they improve readability; avoid repetitive transitions and stock phrases such as "it's worth noting", "delve", "leverage", and "Bottom line".
- Provide brief progress updates during sustained work, focusing on findings, decisions, and remaining uncertainty.
- Report what changed, what was verified, and any remaining uncertainty. Distinguish checks actually run from checks merely recommended.

## Repository Context

- This project trains and evaluates local TabFM, raceformer, and XGBoost racing models. Read the relevant sections of `README.md` and, when needed, `Model.md` before changing a pipeline.
- Core implementation lives in `src/`; split and eligibility utilities live in `tabfm_split/`; dataset views live in `sql/`. Tests live in `tests/` and `src/model/tests/`.
- Names such as `gpt_pick_v2` can identify local feature manifests or experiments. Inspect their use before treating them as external API model identifiers.
- Keep changes focused on the requested behavior. Reuse existing configuration, feature, and model utilities before introducing another implementation.

## Data & Model Integrity

- Preserve chronological train, validation, and test boundaries, complete-race grouping, and competition-aware causal context. Features and historical context must be available at the prediction time of the target workflow.
- Keep sealed test cohorts out of feature selection, hyperparameter tuning, and model selection. Clearly label exploratory analysis that uses test outcomes.
- Preserve the documented market-free deployment default. Current-race market features belong only in explicitly selected market-aware workflows; do not silently promote a diagnostic benchmark to deployment.
- Preserve feature order, preprocessing, label definitions, split manifests, and checkpoint compatibility. If a requested change breaks an existing artifact contract, explain the impact and make the incompatibility explicit.
- For model comparisons, report the cohort, split, metric, and relevant configuration. Do not claim an improvement based only on training performance or incomparable evaluations.

## Workspace & Runtime Care

- Inspect the working tree before editing. Preserve unrelated user changes; do not reset or overwrite them to simplify the task.
- Treat databases, checkpoints, predictions, and experiment reports as user artifacts. Avoid overwriting them during diagnostics or tests; use temporary or clearly separate outputs where practical.
- Check for active training processes before starting a long or memory-intensive run. Do not start concurrent trainers or background jobs without an intentional reason within the authorized task.
- Use the project's documented Python environment when available. Match CPU workers and memory use to the task and current machine load.
- Do not start full training, large tuning sweeps, or database rebuilds merely to verify a small code change. Run them when the task requires them and the scope is authorized.
- Keep credentials out of code, logs, command output, and commits.

## Verification

- Match verification to the scope and impact of the change. Complete required checks; expand testing when a concrete unresolved concern justifies it.
- Prefer focused existing tests for the affected behavior. Add regression coverage when it checks a meaningful failure mode or contract; avoid tests that merely repeat the implementation.
- For data or ranking changes, check the relevant chronology, race grouping, label validity, feature availability, and output contracts.
- For documentation-only changes, inspect the content and diff; model training is unnecessary.
- If verification cannot run, state the exact blocker and what remains unverified. Do not present unexecuted checks or hypothetical model results as evidence.

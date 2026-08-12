# TabFM race model

This checkout trains a binary runner-level `top3_mask` model with complete-race
ranking losses and causal race context. The current recommended architecture
combines:

- TabFM in-context learning;
- optional within-race self-attention;
- strictly earlier races from the same competition for every query race; and
- positive and negative prototypes built directly from normalized historical
  runner features.

## Environment

Activate the PyTorch environment and enter this checkout:

```bash
source /home/theo/perplex/x7/x9/.venv/bin/activate
cd /home/theo/yy1
```

Before starting a long run, confirm that another trainer is not already active:

```bash
pgrep -af 'python.*train_model.py'
jobs -l
```

Do not append `&` unless a deliberate background job is wanted. Running two
trainers at once can exhaust memory and cause Linux to kill one of them.

## Feature manifest

[`tabfm_features.json`](tabfm_features.json) is the source of truth for scratch
training:

- `features` defines the ordered model inputs.
- `zeroed_features` defines inputs retained in that schema but forced to zero.
- Active inputs are `features` minus `zeroed_features`.

The training CLI does not need a separate zero-feature argument. Audit the
manifest against the numeric columns in `race_runners` with:

```bash
python audit_production_features.py
```

To add every newly discovered eligible numeric column to `features` and place
everything not explicitly active into the zero bucket:

```bash
python audit_production_features.py --write
```

Identifiers, timestamps, targets, outcome columns, raw text, and other columns
listed in `IGNORED_COLUMNS` are not feature candidates.

## Recommended scratch training

Use a new output path for a genuinely new model. If that path already exists,
the trainer resumes it unless `--overwrite-existing` is supplied.

```bash
python train_model.py \
  --features-json /home/theo/yy1/tabfm_features.json \
  --output /home/theo/yy1/outputs/3_feature_prototype.pt \
  --epochs 2 \
  --auto-race-schedule \
  --query-races-per-step 5 \
  --learning-rate 0.0001 \
  --early-stopping-patience 3 \
  --race-context-mode self_attention \
  --encode-races-before-icl \
  --context-prototype-branch \
  --context-prototype-dim 16 \
  --context-prototype-max-correction 0.25 \
  --context-prototype-loss-weight 0.25 \
  --label-context-branch \
  --label-context-heads 2 \
  --label-context-max-correction 0.25 \
  --label-context-labels-in-values-only \
  --label-context-loss-weight 0.25 \
  --seed 42 \
  --device cpu \
  --classification-loss-weight 1.0 \
  --pairwise-loss-weight 0.25 \
  --attention-delta-pairwise-loss-weight 0.05 \
  --cardinality-loss-weight 0.0 \
  --context-dependence-loss-weight 0.1 \
  --context-dependence-margin 0.005
```

The startup table should contain:

```text
Context prototype branch         | True
Context prototype source         | Normalized input features
Context prototype dimension      | 16
Context prototype max correction | 0.25
Context prototype direct loss    | 0.25
Label-aware context branch       | True
Label-aware context heads        | 2
Label-aware context direct loss  | 0.25
```

The input-feature prototype architecture is incompatible with the obsolete
hidden-representation prototype checkpoint. Start from scratch with a new path
or `--overwrite-existing` when moving to this architecture.

## Training parameter reference

This section covers every option currently accepted by `train_model.py`.
Values described as inherited are read from `--resume-model` when that option
is supplied. Boolean options using the `--no-...` form explicitly disable the
corresponding feature.

### Data, manifests, and checkpoints

| Parameter | Default | Detailed behavior |
|---|---|---|
| `--db PATH` | `db/race_runners.sqlite` | SQLite source used to resolve the training and validation views, race chronology, competition membership, and market baseline columns. Use the database whose split flags belong to the experiment. |
| `--training-csv PATH` | `outputs/training_records.csv` | Destination for the fresh export of the training view. Training reloads this CSV rather than continuing directly from SQLite, making the exact records inspectable. |
| `--validation-csv PATH` | `outputs/validation_records.csv` | Destination for the fresh validation export. Validation rows and the market baseline are reloaded from this snapshot. It must remain disjoint from training races. |
| `--features-json PATH` | `tabfm_features.json` for scratch training | Ordered feature and zero-bucket manifest. Omit it when resuming so the checkpoint's exact feature order, zeroed features, preprocessing statistics, and prototype input width are inherited. An explicitly incompatible resume manifest is rejected. |
| `--context-json PATH` | `tabfm_context.json` | Context manifest. Its race count supplies the default `--context-races-per-step`; actual query context is still restricted to strictly earlier complete races from the same competition. |
| `--split-manifest PATH` | none | Split-v2 provenance manifest for a clean experiment. Supply the manifest associated with the database split so training can record and validate the split provenance. |
| `--output PATH` | `outputs/tabfm_race_top3.pt` | Checkpoint bundle written after training. If this file already exists, it is automatically treated as a resume source unless `--overwrite-existing` is used. |
| `--resume-model PATH` | none | Loads model weights, feature order, preprocessing, architecture, and relevant metadata from an existing bundle. A fresh AdamW optimizer is created; optimizer momentum is not resumed. |
| `--overwrite-existing` | false | Starts from scratch even if `--output` already exists. This is destructive to the named output, so use a new path unless replacement is deliberate. It cannot be combined with `--resume-model`. |
| `--allow-in-place-fine-tune` | false | Permits `--resume-model` and `--output` to name the same file. Without it, in-place replacement is rejected so the source checkpoint remains available for comparison and recovery. |

### Fine-tuning scope

| Parameter | Default | Detailed behavior |
|---|---|---|
| `--fine-tune-scope SCOPE` | `full_model` from scratch; `icl_and_race_head` for a resumed self-attention model | Selects exact trainable modules. `attention_head_only` trains only `race_set_head`; `decoder_and_race_head` trains the ICL decoder and race head; `icl_and_race_head` trains the full ICL predictor and race head; `icl_race_and_label_context` additionally trains the explicit label-context branch; `label_context_only` trains only that branch; `race_aware_full` also trains the pre-ICL encoder and enabled auxiliary heads; `full_model` updates everything. Every trainable parameter name and trainable/frozen/total count is printed before training. |
| `--fine-tune-attention-head-only` | false | Shortcut for the legacy attention-head-only mode. It requires a resumed self-attention checkpoint and conflicts with any different explicit `--fine-tune-scope`. Prefer the scope option in new commands. |

### Epoch and race scheduling

| Parameter | Default | Detailed behavior |
|---|---|---|
| `--epochs N` | `10` | Maximum complete passes through the generated optimizer-step schedule. Early stopping may finish sooner; the best accepted state, not necessarily the last epoch, is saved. |
| `--steps-per-epoch N` | `100` | Optimizer steps in each epoch when automatic scheduling is disabled. It is derived from the eligible race count when `--auto-race-schedule` is enabled. |
| `--auto-race-schedule` | false | Derives steps per epoch and the effective query-race batch size from eligible complete races. `--query-races-per-step` becomes the maximum target batch size. This is recommended for full-dataset training. |
| `--context-races-per-step N` | number of races in `--context-json` | Number of most-recent, complete, strictly earlier same-competition races assigned independently to each query. Increasing it changes the context distribution and should be validated rather than assumed beneficial. |
| `--query-races-per-step N` | `80` | Maximum complete query races in one optimizer step. Smaller values reduce memory use and give more updates per epoch; the prototype runs use `5`. |
| `--print-race-schedule` | false | Prints `race_id:race_number` for context and query races in chronological step order. Enable it when auditing causal membership or investigating unexpected scheduling. |
| `--debug-training-output` | false | Prints the full legacy diagnostics: parameter table, every optimizer batch, context ablations, all validation cohorts, and progress-race rankings. Without it, training prints concise setup, probe, epoch, trend, and checkpoint summaries. |
| `--batch-rows N` | `256` | Deprecated compatibility argument. Current batches are composed of complete races and therefore have variable row counts; this value does not control the active race scheduler. |
| `--min-race-number N` | none | Restricts optimizer-step context and query pools to races whose `race_number` is at least `N`. Validation and its fixed context are unchanged, so this is an experimental training filter rather than a validation filter. |

### Optimizer and numerical controls

| Parameter | Default | Detailed behavior |
|---|---|---|
| `--learning-rate RATE` | `0.0003` from scratch; `0.00003` when resuming | AdamW step size. Fine-tuning uses the lower default to avoid destroying useful weights. Values above `0.00003` on resume are rejected unless the high-rate safety override is supplied. |
| `--allow-high-fine-tune-learning-rate` | false | Bypasses the resumed-model learning-rate ceiling. Use only for a controlled destructive-update experiment with a separate output path. |
| `--weight-decay RATE` | `0.0001` | AdamW decoupled weight decay. It regularizes trainable weights; excessive values can suppress the relatively small context corrections. |
| `--max-grad-norm VALUE` | `1.0` | Clips the total gradient norm before each optimizer update. It must be positive and limits unstable steps without changing the forward loss. |
| `--seed N` | `42` | Seeds Python, NumPy, Torch, and the training sampler. Reusing it makes schedule and probe comparisons reproducible; use additional seeds when estimating result variance. |
| `--device DEVICE` | `cuda` when available, otherwise `cpu` | Torch execution device, such as `cpu`, `cuda`, or `cuda:0`. CPU is slower but avoids GPU availability and memory constraints. |

### Validation, probes, and checkpoint selection

| Parameter | Default | Detailed behavior |
|---|---|---|
| `--early-stopping-patience N` | `3` | Stops after `N` consecutive eligible epochs fail to improve checkpoint-selection metrics. It must be positive. Small chronological cohorts disable patience unless explicitly allowed. |
| `--allow-small-cohort-early-stopping` | false | Enables patience-based stopping when the `chronological_representative` cohort contains fewer than 20 complete races. This makes selection more sensitive to noise. |
| `--probe-every-steps N` | `10` | Runs deterministic `FIXED_PROBE` and context-ablation diagnostics every `N` optimizer updates. Lower values give faster feedback but increase runtime. |
| `--probe-races N` | `20` | Number of complete development races held fixed for deterministic probes. It must be positive; a larger cohort is steadier but more expensive. |
| `--step-loss-window N` | `10` | Number of recent optimizer-step losses used in `rolling_loss`. It smooths logging only and does not change gradients. |
| `--max-valid-races N` | none | Compatibility check only. Validation is never truncated; a value below the complete flagged validation cohort is rejected. |
| `--stress-top3-recall-max-drop VALUE` | `0.5` | Checkpoint-selection guardrail in the range `[0,1]`. It limits the permitted absolute top-three-recall degradation on the market-miss stress cohort relative to its best observed result. |
| `--progress-race-id ID` | first validation race | Chooses one race whose runner ranking is printed after every epoch. This is a human-readable diagnostic and does not select the checkpoint. |
| `--valid-frac VALUE` | `0.20` | Deprecated. The active partition comes from `race_runners.is_validation`; changing this value does not create a random validation split. |
| `--train-cutoff-iso TIMESTAMP` | none | Deprecated and rejected by the flag-based validation pipeline. Set chronological membership in the database split instead. |

### Race-context and prototype architecture

| Parameter | Default | Detailed behavior |
|---|---|---|
| `--race-context-mode {none,self_attention}` | `none` from scratch; inherited on resume | Enables interaction between runners in the same query race. `self_attention` lets runner representations influence one another before the final ranking correction. |
| `--race-context-dim N` | `32` | Hidden width of the within-race attention branch. It must be positive and divisible by `--race-context-heads`. Larger values increase capacity and compute. |
| `--race-context-layers N` | `1` | Number of stacked within-race attention layers. More layers add capacity and latency and require scratch training or an architecture-compatible checkpoint. |
| `--race-context-heads N` | `2` | Number of attention heads. It must be positive and divide the context dimension exactly. |
| `--race-context-ff-dim N` | `64` | Hidden width of the race-context feed-forward sublayer. Larger values increase parameters and CPU cost. |
| `--race-context-residual` / `--no-race-context-residual` | enabled | Controls whether the race-head correction is added residually to the base logits. Resumed models inherit their stored setting when neither form is supplied. |
| `--race-head-scale VALUE` | `1.0` from scratch; inherited on resume | Multiplies the post-ICL race-head logit residual before it is added to the base logits. `1.0` reproduces existing checkpoints; use held-out scale sweeps such as `0,0.1,0.25,0.5,0.75,1` to measure whether the branch improves ranking rather than assuming it does. |
| `--label-context-temperature VALUE` | `1.0` from scratch; inherited on resume | Divides the explicit label-context attention logits. Values below `1` sharpen retrieval; values above `1` flatten it. This does not alter the main ICL mechanism. |
| `--label-context-top-k K` | `0` from scratch; inherited on resume | Applies top-k masking independently to each label-context head and query runner before softmax. `0` disables masking; padded runners are always excluded. |
| `--checkpoint-metric METRIC` | `composite` | Selects checkpoints by `loss`, `top3_recall`, `ndcg3`, or the logged lexicographic composite `(top3_recall, ndcg3, exact_top3_set, pairwise_ranking_accuracy, -logloss)`. |
| `--encode-races-before-icl` / `--no-encode-races-before-icl` | disabled from scratch; inherited on resume | Applies the representation-level race encoder before the ICL predictor. It changes the architecture and therefore requires compatible weights or scratch retraining. |
| `--context-prototype-branch` / `--no-context-prototype-branch` | disabled from scratch; inherited on resume | Enables the positive/negative historical feature prototypes and bounded query-logit correction. A prototype checkpoint cannot be resumed with this branch disabled. |
| `--context-prototype-dim N` | `16` | Width of the learned prototype metric space. It must be positive and is part of checkpoint tensor compatibility. |
| `--context-prototype-max-correction VALUE` | `0.5` | Positive bound on the absolute per-class logit correction contributed by prototypes. The recommended models explicitly use the more conservative value `0.25`. |

### Loss weights and context objectives

All loss weights must be non-negative. A value of zero disables that component.
The total objective is the weighted sum of the enabled components.

For compatibility with older commands, the loss flags also accept these exact
underscore aliases: `--classification_loss_weight`,
`--auxiliary_row_loss_weight`, `--pairwise_loss_weight`,
`--attention_delta_pairwise_loss_weight`,
`--context_prototype_loss_weight`, `--label_context_loss_weight`,
`--cardinality_loss_weight`,
`--context_dependence_loss_weight`, and `--context_dependence_margin`. They are
identical to the hyphenated forms documented below; do not supply both forms in
one command.

| Parameter | Default | Detailed behavior |
|---|---|---|
| `--classification-loss-weight VALUE` | `1.0` | Weight on class-weighted binary top-three cross-entropy, averaged within each race and then equally across races. This is the primary probability-learning objective. |
| `--pairwise-loss-weight VALUE` | `0.0` | Weight on the softplus ranking loss between every top-three and non-top-three runner within each complete query race. The recommended prototype runs use `0.25`. |
| `--attention-delta-pairwise-loss-weight VALUE` | `0.0` | Applies pairwise ranking loss directly to the self-attention race-head correction, ensuring that branch learns ranking information rather than letting base logits do all the work. It requires self-attention; the recommended value is `0.05`. |
| `--context-prototype-loss-weight VALUE` | `0.25` when the prototype branch is enabled, otherwise `0` | Applies pairwise ranking loss directly to prototype-only query corrections. This is the direct supervision that connects historical labels and runner features to rankings. |
| `--label-context-loss-weight VALUE` | `0.25` when label-aware context is enabled, otherwise `0` | Applies pairwise ranking loss directly to the historical cross-attention correction so the branch must learn within-race ranking rather than only confidence shifts. |
| `--cardinality-loss-weight VALUE` | `0.0` | Penalizes the squared difference between the sum of runner top-three probabilities and three, normalized by three, within each race. Leave it at zero when independent binary probabilities are preferred. |
| `--context-dependence-loss-weight VALUE` | `0.1` | Weight on the contrastive context objective comparing correct context with a deterministic label-permuted version. It teaches correct context to outperform incorrect label-feature associations. |
| `--context-dependence-margin VALUE` | `0.02` | Required loss advantage of correct context over permuted context. Once the advantage reaches the margin, the context-dependence penalty becomes zero. Prototype runs use the gentler value `0.005`. |
| `--auxiliary-row-loss-weight VALUE` | `0.0` | Reserved CLI compatibility option. It is currently parsed and validated but is not added to the implemented training objective; leave it at zero until the code supplies the documented auxiliary loss. |

### Experimental and leakage-prone controls

| Parameter | Default | Detailed behavior |
|---|---|---|
| `--fine-tune-race-id ID` | none | Experiment only. Moves exactly one complete race into the optimizer pool during resume and deliberately uses its labels. Results on that race are no longer out-of-sample. |
| `--classroom-overfit-all-races` | false | Experiment only. Uses every exposed complete race for both optimization and validation, including fixed context races. This intentionally leaks labels and must never be used for production evaluation. |

For the authoritative parser defaults after future code changes, run:

```bash
python train_model.py --help
```

## Causal context contract

Each training and validation query race receives its own independent sequence.
Its context consists of the most recent configured number of complete races
that satisfy both conditions:

1. The context race belongs to the same `competition_id` as the query.
2. Its timestamp is strictly earlier than the query timestamp.

The number of context races defaults to the race count in
`tabfm_context.json`; override it with `--context-races-per-step`. Queries with
insufficient earlier same-competition context are rejected rather than filled
with future races.

Before preprocessing, the selected SQLite views are exported to and reloaded
from:

- `outputs/training_records.csv`
- `outputs/validation_records.csv`

Use `--training-csv` and `--validation-csv` for different snapshot paths.

## Context prototype branch

For every independently contextualized query sequence, the branch:

1. Projects normalized historical runner features into a learned metric space.
2. Averages label-0 runners into a negative prototype.
3. Averages label-1 runners into a positive prototype.
4. Compares every query runner with both prototypes.
5. Adds a symmetric bounded correction to its ordinary binary logits.

If either historical label class is absent, the correction fails closed to
zero. `--context-prototype-loss-weight` directly applies pairwise ranking loss
to prototype-only query corrections. It defaults to `0.25` when the branch is
enabled and to zero otherwise.

## Label-aware historical cross-attention

Enable the learned-representation context path with:

```bash
--label-context-branch \
--label-context-heads 2 \
--label-context-max-correction 0.25 \
--label-context-labels-in-values-only \
--label-context-loss-weight 0.25
```

Each query runner representation acts as an attention query. Historical runner
representations alone act as keys, so feature similarity chooses which examples
to retrieve. Historical representations plus a learned `top3_mask` embedding
act as values, so outcomes affect the retrieved information without shortcutting
the similarity search. The resulting query-only correction is bounded before it
is added to the final logits. Query labels are never read.
The branch supports full forward training and checkpoint prefill/decode caching.
Existing checkpoints inherit the branch as disabled, but it can be added during
full-model fine-tuning by explicitly passing `--label-context-branch` and saving
to a new output path. Label-context checkpoints created before this retrieval
change retain their legacy behavior unless explicitly upgraded with
`--label-context-labels-in-values-only` and retrained.

## Current prototype models

The current prototype checkpoints use the normalized-input feature prototype
branch described above. They differ primarily in the number of races exposed
during training:

| Checkpoint | Training scope |
|---|---:|
| `outputs/3_prototype.pt` | 200 races |
| `outputs/4_prototype.pt` | 1,000 races |
| `outputs/5_prototype.pt` | All 8,058 eligible training races available at training time |

The saved `5_prototype.pt` bundle records the following effective model and
training configuration:

- 176 ordered manifest inputs, of which 123 are retained in the zero bucket;
- nine strictly earlier same-competition context races per query;
- five query races per optimizer step;
- within-race self-attention with race encoding before ICL;
- a 16-dimensional normalized-input prototype branch with a maximum correction
  of `0.25`;
- full-model scratch training, with epoch 2 retained as the best epoch.

On 9 August 2026, the three checkpoints were evaluated on the same held-out,
previously untrained races from `competition_id=590`. The command requested the
latest 100 races; 50 targets were available and scored, with 48 satisfying the
complete-race metric contract:

| Checkpoint | Top-3 recall | Exact top-3 | Contained top-4 | Contained top-5 | Contained top-6 | ROC AUC | Logloss | Seconds/target |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `3_prototype.pt` | 0.5278 | 0.0625 | 0.2083 | 0.4167 | 0.5625 | 0.7720 | 0.6973 | 0.219 |
| `4_prototype.pt` | 0.5556 | 0.1042 | 0.2708 | 0.4167 | 0.6042 | 0.7859 | 0.6858 | 0.244 |
| `5_prototype.pt` | **0.5833** | **0.1042** | 0.2292 | **0.4375** | 0.5417 | **0.8056** | **0.5441** | 0.247 |

These results show improving held-out top-three recall and ROC AUC as the
training set grows. `5_prototype.pt` is the current recommended source
checkpoint because it has the strongest top-three recall, ROC AUC, and
logloss. Recheck this conclusion on additional untouched competitions rather
than treating one 48-race cohort as permanent model selection evidence.

## Training diagnostics

`pre_update_training_batch` is a noisy batch-level measurement taken before
`optimizer.step()`. Use the deterministic `FIXED_PROBE` and
`CONTEXT_ABLATION_PROBE` lines for progress decisions.

The context probe evaluates the same races using:

- correct context labels;
- deterministically permuted labels;
- zeroed labels; and
- flipped binary labels.

Prototype-specific fields are:

- `prototype_pairwise_loss`: random ordering is approximately `0.69315`; it
  should generally move below that value.
- `prototype_abs_mean`: magnitude of the bounded prototype correction.
- `prototype_permutation_delta`: change in the prototype correction after
  reassigning historical labels to different runners.
- `permuted_minus_correct_loss`: positive values mean correct context performs
  better on that training batch.

A healthy run should show non-zero prototype magnitudes, material prototype
permutation deltas, and worse fixed-probe validation results with permuted
context. At step 100 of the first successful input-feature prototype run, the
fixed probe reported:

```text
correct:  top3_recall=0.6167 auc=0.7946 logloss=0.69698
permuted: top3_recall=0.5000 auc=0.6658 logloss=0.70604
```

This is development-probe evidence, not final held-out performance.

Stop and review the run after repeated probes if correct AUC and top-three
recall fall while correct logloss rises, or if all of the following return:

- `prototype_pairwise_loss` remains at approximately `0.69315`;
- `prototype_abs_mean` is effectively zero;
- `prototype_permutation_delta < 0.0001`; and
- permuted context performs as well as correct context.

Do not stop based on one noisy `pre_update_training_batch` line. Complete at
least the current epoch when safe because the best state is retained in memory
and the final checkpoint bundle is written after the training loop.

## Fine-tuning

Keep the source checkpoint unchanged and use `3e-5` or less. Omit
`--features-json` so the exact saved feature order, zero bucket, preprocessing,
and prototype input dimension are inherited:

```bash
python train_model.py \
  --resume-model /home/theo/yy1/outputs/5_prototype.pt \
  --output /home/theo/yy1/outputs/5_prototype_fine.pt \
  --epochs 8 \
  --auto-race-schedule \
  --query-races-per-step 5 \
  --learning-rate 0.00003 \
  --early-stopping-patience 4 \
  --fine-tune-scope icl_and_race_head \
  --seed 42 \
  --device cpu \
  --classification-loss-weight 1.0 \
  --pairwise-loss-weight 0.25 \
  --attention-delta-pairwise-loss-weight 0.05 \
  --context-prototype-loss-weight 0.25 \
  --cardinality-loss-weight 0.0 \
  --context-dependence-loss-weight 0.1 \
  --context-dependence-margin 0.005
```

Fine-tuning above `3e-5` is rejected unless
`--allow-high-fine-tune-learning-rate` is explicitly supplied. Using the same
path for `--resume-model` and `--output` is rejected unless
`--allow-in-place-fine-tune` is supplied.

Patience-based early stopping is disabled when the
`chronological_representative` checkpoint-selection cohort has fewer than 20
complete races, unless `--allow-small-cohort-early-stopping` is deliberately
provided.

## Single-race prediction

Predict one race using all eligible earlier finished races from its competition
in addition to the checkpoint context:

```bash
python predict_race.py \
  --models-dir /home/theo/yy1/outputs \
  --device cpu \
  --race-id 10785717 \
  --include-competition-history-context
```

The predictor prints the selected context strategy and race IDs. It also warns
when the inference context window exceeds the context size used in training.

## Single-race model debugger

Explain one checkpoint's prediction runner by runner:

```bash
python debug_race.py \
  --checkpoint outputs/classroom_competition_590_overfit.best-epoch-001.pt \
  --race-id 10147278 \
  --device cpu
```

The report shows the exact same-competition chronological context, preprocessing
outliers, pre-ICL field-encoder movement, and every additive logit transition:
base, label context, raw and scaled race-head residual, prototype, and final
score. It also reports market and actual results, context-label
counterfactuals, stage-by-stage top-three accuracy, the stage at which a correct
runner was displaced, probability cardinality, controlled feature-family
ablations, integrated-gradients attribution, branch influence, attention
entropy/effective retrieval count, and runner-order invariance. Use
`--no-base-attribution` or `--no-context-ablation` for a faster report, or
`--output-csv path.csv` to save the runner stage table.

For label-context checkpoints, Stage 4 reports every attention head separately,
historical-runner and historical-race effective counts, per-race attention mass,
positive-label lift, query-to-query cosine/JS/Jaccard overlap, projection norms,
and the branch output-projection norm. `--debug-attention-details` prints full
race-mass and pairwise matrices. The debugger also runs label-context-only
uniform/top-5/top-10/top-20 counterfactuals and an inference-only temperature
sweep (default `0.25,0.5,1,2`). An old
checkpoint such as `outputs/1.pt` correctly reports this branch as `OFF` and a
zero `LabelCtx Δp`; debug a checkpoint produced by a run using
`--label-context-branch` to inspect the new path.

Evaluate branch usefulness across the exposed race cohort before changing the
architecture:

```bash
python evaluate_model_stages.py \
  --checkpoint outputs/model.pt \
  --race-source checkpoint_validation \
  --race-head-scales 0,0.1,0.25,0.5,0.75,1 \
  --summary-json outputs/model_stage_summary.json \
  --output-csv outputs/model_stage_ablation.csv
```

This reports all eight additive branch combinations, sequential stage improve/
degrade counts, NDCG@3, pairwise ranking accuracy, race-head usefulness,
cardinality distributions, per-head retrieval quality, and inference-only
market/race-relative feature ablations. A validation cohort that overlaps the
embedded training manifest fails unless `--allow-evaluation-leakage` is passed;
the override remains visibly watermarked. A classroom checkpoint is
explicitly marked as provenance-unsafe: if its exact training race manifest was
not embedded, a later database view may mix seen and unseen races.
New checkpoints embed their training/query and validation race IDs; use
`checkpoint_validation` so later view changes cannot silently change the target
cohort. They also record SHA-256 hashes for the exported training and validation
CSVs; keep those snapshots immutable when exact context-row reproducibility is
required.

Audit the exact checkpoint preprocessing without modifying it:

```bash
python audit_standardized_features.py \
  --checkpoint outputs/model.pt \
  --training-csv outputs/training_records.csv \
  --validation-csv outputs/validation_records.csv \
  --output-csv outputs/model_feature_scaling_audit.csv
```

Any proposed transform from this audit must become a versioned preprocessing
contract fitted on training data and reused unchanged for validation and live
prediction.

For the controlled label-context A/B/C experiment, keep the split, seed,
features, optimiser, schedule, losses, context size, and `--race-head-scale 0.1`
identical. Change only these arguments:

```text
A: --label-context-branch --label-context-temperature 1 --label-context-top-k 0
B: --no-label-context-branch --label-context-loss-weight 0
C: --label-context-branch --label-context-temperature <chosen-from-diagnostics>
   --label-context-top-k <chosen-from-diagnostics>
```

Choose C from clean-validation ranking evidence, not from a desired effective
retrieval count alone. Evaluate each saved checkpoint with
`--race-source checkpoint_validation`; compare the stage and retrieval tables.
No evaluator performs training or modifies a checkpoint.

After producing summaries for all three models, build the requested clean
comparison table with:

```bash
python compare_label_context_reports.py \
  --model A=outputs/A_summary.json \
  --model B=outputs/B_summary.json \
  --model C=outputs/C_summary.json \
  --output-csv outputs/label_context_abc.csv
```

## Prediction backtests

Restrict backtest targets to one competition and score the most recent 20
matching finished races chronologically:

```bash
python predict_race.py \
  --models-dir /home/theo/yy1/outputs \
  --device cpu \
  --backtest \
  --backtest-max-races 20 \
  --competition-id 590
```

Progress output reports inference time and average seconds per target. The
final metrics table includes `inference_seconds`, `seconds_per_target`, and
`checkpoint_seconds`; the final completion line reports total time across all
checkpoints.


python finetune_raceformer.py \
    --checkpoint outputs/raceformer_11m.pt \
    --output outputs/raceformer_12m.pt \
    --scope full \
    --features-json tabfm_features.json \
    --layoff-bucket-mode none \
    --learning-rate 3e-6 \
    --epochs 30 \
    --save-strategy source_guarded \
    --device cpu

      python finetune_raceformer.py \
      --checkpoint outputs/raceformer_competition_mf.pt \
      --output outputs/raceformer_competition_mf1.pt \
      --training-competition-id 570,335,580,351,488,279,231,330,207,366,585 \
      --validation-competition-id 317,340,410,638,602,505,588,635,520 \
      --scope head_only \
      --features-json tabfm_features.json \
      --layoff-bucket-mode none \
      --learning-rate 3e-6 \
      --races-per-batch 32 \
      --epochs 15 \
      --early-stopping-patience 5 \
      --save-strategy source_guarded \
      --device cpu

### Retrain RaceFormer with robust current-price preprocessing

Scratch RaceFormer checkpoints now use preprocessing contract v3: `open_price`,
`fluc1`, and always-available `fluc2` are signed-`log1p` transformed, robustly
scaled, clipped to `+/-5`, and accompanied by within-race percentile features.
`competition_id` and `race_number` are neutralized by `tabfm_features.json`.

```bash
python train_raceformer.py \
  --no-export \
  --training-csv outputs/raceformer_training.csv \
  --validation-csv outputs/raceformer_validation.csv \
  --features-json tabfm_features.json \
  --output outputs/raceformer_v3.pt \
  --training-competition-id 570,335,580,351,488,279,231,330,207,366,585 \
  --validation-competition-id 317,340,410,638,602,505,588,635,520 \
  --chronological-validation-races 0 \
  --standardized-clip 5 \
  --races-per-batch 32 \
  --epochs 40 \
  --early-stopping-patience 8 \
  --checkpoint-metric composite \
  --device cpu
```

The run prints the held-out `fluc2` market baseline before training and a final
`deployment_gate_model_beats_market=yes|no`. Do not promote a checkpoint whose
gate is `no`.

If the unconstrained v3 model does not beat that gate, run the market-anchored
residual experiment. It fits a monotonic anchor from the within-race `fluc2`
percentile on training races only, starts with zero residual (therefore the
exact market ranking), and regularizes all learned logit corrections toward zero:

```bash
python train_raceformer.py \
  --no-export \
  --training-csv outputs/raceformer_training.csv \
  --validation-csv outputs/raceformer_validation.csv \
  --features-json tabfm_features.json \
  --output outputs/raceformer_market_residual_v3.pt \
  --variant market_residual \
  --market-residual-scale 0.25 \
  --market-residual-weight 0.05 \
  --training-competition-id 570,335,580,351,488,279,231,330,207,366,585 \
  --validation-competition-id 317,340,410,638,602,505,588,635,520 \
  --chronological-validation-races 0 \
  --standardized-clip 5 \
  --races-per-batch 32 \
  --epochs 60 \
  --early-stopping-patience 12 \
  --checkpoint-metric composite \
  --seed 42 \
  --device cpu
```

Epoch zero is retained as the exact market-ranking fallback. A trained epoch is
saved only when its held-out composite exceeds that fallback; promotion still
requires `deployment_gate_model_beats_market=yes`.


python finetune_raceformer.py \
      --checkpoint outputs/raceformer_competition.pt \
      --output outputs/raceformer_competition_x1.pt \
      --training-competition-id 570,335,580,351,488,279,231,330,207,366,585 \
      --validation-competition-id 317,340,410,638,602,505,588,635,520 \
      --scope head_only \
      --features-json tabfm_features.json \
      --layoff-bucket-mode none \
      --learning-rate 3e-6 \
      --races-per-batch 32 \
      --epochs 15 \
      --early-stopping-patience 5 \
      --save-strategy source_guarded \
      --device cpu

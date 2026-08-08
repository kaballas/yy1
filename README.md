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
```

The input-feature prototype architecture is incompatible with the obsolete
hidden-representation prototype checkpoint. Start from scratch with a new path
or `--overwrite-existing` when moving to this architecture.

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

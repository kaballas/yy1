# TabFM Learning and Evaluation Todo List

The learning-rate correction is the starting point. Complete the following
items in order; do not use the final holdout to choose settings.

## 0. Establish the low-rate reference run

- [ ] Preserve `outputs/tabfm_race_top3.pt` unchanged as the baseline.
- [ ] Fine-tune into a new output file; never use the resume checkpoint as the
      output while experimenting.
- [ ] Start with full-model AdamW at `3e-5` and reduce to `1e-5` or `2e-6` if
      fixed-development validation deteriorates over a complete epoch.
- [ ] Use `--auto-race-schedule` so each eligible query race is normally seen
      once per epoch instead of using 8,000 query slots for 3,779 races.
- [ ] Use at least five epochs of early-stopping patience because the discrete
      whole-race metrics move in runner-sized increments.
- [ ] Record the command, seed, source checkpoint SHA-256, database identity,
      view SQL, split-manifest hash, and resulting metrics.

Initial command:

```bash
python train_model.py \
  --resume-model /home/theo/yy1/outputs/tabfm_race_top3.pt \
  --output /home/theo/yy1/outputs/tabfm_race_top3_lr3e5.pt \
  --epochs 20 \
  --auto-race-schedule \
  --context-races-per-step 10 \
  --learning-rate 0.00003 \
  --early-stopping-patience 5 \
  --min-race-number 5 \
  --race-context-mode self_attention \
  --pairwise-loss-weight 1.0 \
  --attention-delta-pairwise-loss-weight 0.5 \
  --cardinality-loss-weight 0 \
  --seed 42 \
  --device cpu
```

Acceptance criteria:

- [ ] Epoch-end fixed-development top-3 recall does not decline.
- [ ] Log loss remains finite and does not make a large jump such as
      `0.557 -> 0.992` after one update.
- [ ] The selected checkpoint is a genuinely improved epoch, not epoch 0.

## 1. Make step logging measure learning correctly

- [x] Stop describing the current step metric as post-update: it is calculated
      from logits produced before `optimizer.step()`.
- [x] Keep batch metrics as diagnostics, but label them
      `pre_update_training_batch` and print the number of complete races.
- [x] Add a deterministic fixed probe set of complete development races.
- [x] Evaluate the same probe races before training and every configurable
      number of optimizer steps.
- [x] Print both raw step loss and a rolling mean; do not expect metrics from
      different random 80-race batches to be monotonic.
- [x] Add a test proving high- and low-learning-rate runs start with identical
      pre-update step-1 metrics.

Acceptance criteria:

- [ ] Every metric line identifies `training_batch`, `fixed_probe`, or
      `full_development`.
- [ ] A fixed-probe comparison uses identical race IDs and context for every
      checkpoint.

## 2. Match training and inference context contracts

- [ ] Measure and persist context race count and context row count for every
      training step and evaluation call.
- [ ] Remove the current large mismatch: training uses about 10 races / 103
      rows while validation uses 123 races / 1,504 rows.
- [ ] Choose one explicit policy:
  - train with context sizes representative of inference; or
  - cap inference and validation context to the distribution used in training.
- [ ] Sample context strictly before each query race.
- [ ] Ensure context and query race IDs are disjoint and preserve whole races.
- [ ] Rebuild any prediction prefill/cache after model weights or context change.

Acceptance criteria:

- [ ] Training, development, backtest, and prediction metadata state the same
      context policy.
- [ ] Tests reject future context, target self-context, partial races, and
      context/query overlap.

## 3. Finish the leakage-safe chronological split

- [ ] Generate and validate a split-v2 manifest using the existing
      `tabfm_split` package.
- [ ] Persist non-overlapping `training`, `development`, and `final_holdout`
      whole-race partitions ordered by `(start_time_iso, race_id)`.
- [ ] Fit median and scale only on the training partition.
- [ ] Require training context to be earlier than its query race.
- [ ] Require all fixed development context to predate every development target,
      or replace it with target-specific chronological context.
- [ ] Fail closed when `--split-manifest` is absent for a production-eligible
      run.
- [ ] Keep `market_miss_stress` orthogonal to chronological partition labels.
- [ ] Do not inspect the final holdout until architecture, features, losses,
      context policy, and checkpoint-selection rules are frozen.

Acceptance criteria:

- [ ] No race ID overlaps partitions.
- [ ] No development/final label is used as context or an optimizer target.
- [ ] Dataset and ordered race-ID hashes are reproducible from a fresh database
      connection.

## 4. Repair and document the SQL data contract

- [ ] Reconcile `--min-race-number` with the installed view, which currently
      hard-codes `race_number > 4`; remove the redundant/conflicting predicate
      or make the threshold explicit in one place.
- [ ] Decide whether holding out competitions `581`, `351`, and `335` is a
      deliberate domain-generalization test or an accidental train/validation
      split.
- [ ] Keep the competition holdout as a separately named cohort if it is
      intentional; do not call it chronological validation.
- [ ] Print and archive exact `sqlite_master` view definitions, emitted loader
      SQL, bindings, `EXPLAIN QUERY PLAN`, and post-SQL whole-race exclusions.
- [ ] Report eligible, invalid, context-excluded, selected, and total race counts.
- [ ] Investigate the 128 invalid training races and six invalid validation races;
      repair labels upstream rather than silently admitting incomplete races.

Acceptance criteria:

- [ ] Every selected race has at least three runners and exactly three
      `top3_mask=1` rows.
- [ ] The effective race-number and competition filters are unambiguous.

## 5. Build trustworthy development cohorts

- [ ] Increase `chronological_representative` from 11 to at least the required
      20 complete races, preferably substantially more.
- [ ] Assign all development races explicitly; remove the 133-race
      `legacy_combined` fallback from checkpoint selection.
- [ ] Populate a real `market_miss_stress` cohort or remove the inactive stress
      command-line setting from experiments.
- [ ] Print checkpoint-selection cohort, race count, and the exact reason a
      candidate was accepted or rejected.
- [ ] Require a minimum evidence threshold before allowing early stopping or
      checkpoint promotion.

Acceptance criteria:

- [ ] `--stress-top3-recall-max-drop` is either demonstrably active or clearly
      reported as inactive.
- [ ] Checkpoint selection never silently falls back to a legacy cohort.

## 6. Make feature neutralization explicit

- [ ] Document that bare `--zero-features` means `zeroed_features=[]` and does
      not zero every feature.
- [ ] Omit `--zero-features` to inherit the resume checkpoint contract.
- [ ] When neutralizing features, list every exact column name and store it in
      checkpoint metadata.
- [ ] Audit direct and derived market features separately, including prices,
      price ranks, movements, consensus, and agreement features.
- [ ] Check every feature for point-in-time availability and target leakage.
- [ ] Reject resume when the requested feature/zeroing contract differs unless
      an explicit experiment override writes to a separate checkpoint.

Acceptance criteria:

- [ ] Startup prints inherited versus explicitly overridden feature masks.
- [ ] A checkpoint cannot silently change its feature-neutralization contract.

## 7. Add market and simple-model promotion gates

- [ ] Compare every checkpoint against direct `fluc2` ranking on exactly the
      same complete races.
- [ ] Add a simple linear/logistic or tree baseline using the same training-only
      preprocessing and chronological partitions.
- [ ] Report top-3 recall, exact top-3 set, contained top 4/5/6, AUC, log loss,
      and runner counts for model and baselines.
- [ ] Report paired per-race wins, losses, and ties against the market.
- [ ] Require the TabFM model to add measurable value over market recall
      `0.5278`, rather than merely improve from its current `0.4282`.

Acceptance criteria:

- [ ] No model is promoted solely because training loss or AUC improved.
- [ ] Promotion requires discrete whole-race improvement with uncertainty
      reported by a paired bootstrap or another predeclared paired test.

## 8. Run controlled architecture and loss ablations

- [ ] Freeze the split, seed, context draws, preprocessing, schedule, and
      development probe before comparing settings.
- [ ] Compare `full_model`, `icl_and_race_head`, and `attention_head_only` scopes.
- [ ] Test one factor at a time: final-logit pairwise weight, direct attention
      delta weight, cardinality weight, and context size.
- [ ] Track gradient norms and parameter-update norms by model component.
- [ ] Stop any run immediately if fixed-probe log loss explodes or non-finite
      values appear.
- [ ] Do not select architecture or loss weights on the final holdout.

Acceptance criteria:

- [ ] Each ablation has a baseline, one changed variable, identical race IDs,
      and a separate output/checkpoint.

## 9. Final OOS evaluation and promotion

- [ ] Freeze code commit, feature manifest, context policy, split manifest,
      checkpoint-selection rule, and chosen hyperparameters.
- [ ] Evaluate the final holdout exactly once after freezing decisions.
- [ ] Compare against `fluc2` and the simple-model baseline on identical races.
- [ ] Run a paired bootstrap over complete races and report uncertainty.
- [ ] Save runner-level predictions, race-level metrics, exclusions, metadata,
      and data hashes.
- [ ] Mark the model production-eligible only if chronology, context, features,
      checkpoint provenance, and market-value gates all pass.
- [ ] Otherwise retain the result as a diagnostic checkpoint and document the
      failed gate.

## Recommended implementation order

- [ ] Step logging and fixed probe.
- [ ] Context-size and temporal-context contract.
- [ ] Split-v2 production enforcement.
- [ ] SQL/view and cohort cleanup.
- [ ] Feature audit and explicit masking contract.
- [ ] Controlled loss/scope ablations.
- [ ] Market comparison and paired uncertainty.
- [ ] One-time final holdout and promotion decision.

# TabFM race-model fine-tuning

Activate the PyTorch environment used by this checkout:

```bash
cd /home/theo/perplex/x7/x9/live
source ../.venv/bin/activate
cd /home/theo/yy1
```

## Recommended fine-tune

Keep the source checkpoint unchanged, use the checkpoint-safe learning rate,
average each optimizer update across multiple complete query races, and cover
the eligible training pool once per epoch:

```bash
python train_model.py \
  --resume-model /home/theo/yy1/outputs/merged_model.pt \
  --output /home/theo/yy1/outputs/2.pt \
  --epochs 8 \
  --auto-race-schedule \
  --query-races-per-step 10 \
  --learning-rate 0.00003 \
  --early-stopping-patience 4 \
  --fine-tune-scope icl_and_race_head \
  --min-race-number 3 \
  --race-context-mode self_attention \
  --seed 42 \
  --device cpu \
  --classification_loss_weight 1.0 \
  --pairwise_loss_weight 0.25 \
  --attention_delta_pairwise_loss_weight 0.05 \
  --cardinality_loss_weight 0.0
```

Training and validation both use the same context contract: every query race
gets its own sequence containing the most recent strictly earlier complete
training races. The number of context races defaults to the race count in
`tabfm_context.json`.

Before model preprocessing or training starts, the selected SQLite views are
exported and then reloaded from these CSV snapshots:

- `outputs/training_records.csv`
- `outputs/validation_records.csv`

Use `--training-csv` and `--validation-csv` to select different paths. Both
files are replaced with fresh, ordered exports on every run; the model arrays,
labels, timestamps, validation flags, and market baseline are loaded from the
CSV files rather than directly from SQLite.

The command intentionally omits `--allow-small-cohort-early-stopping`. Until
the `chronological_representative` cohort contains at least 20 complete races,
all requested epochs run and checkpoint selection still retains the best
chronological epoch.

Fine-tuning above `3e-5` is rejected unless
`--allow-high-fine-tune-learning-rate` is explicitly supplied. Explicitly
using the same path for `--resume-model` and `--output` is also rejected unless
`--allow-in-place-fine-tune` is supplied.




python train_model.py \
  --output /home/theo/yy1/outputs/base.pt \
  --epochs 2 \
  --auto-race-schedule \
  --query-races-per-step 10 \
  --learning-rate 0.00003 \
  --early-stopping-patience 4 \
  --min-race-number 3 \
  --race-context-mode self_attention \
  --seed 42 \
  --device cpu \
  --classification_loss_weight 1.0 \
  --pairwise_loss_weight 0.25 \
  --attention_delta_pairwise_loss_weight 0.05 \
  --cardinality_loss_weight 0.0


python train_model.py \
  --resume-model /home/theo/yy1/outputs/merged_model.pt \
  --output /home/theo/yy1/outputs/2.pt \
  --epochs 8 \
  --auto-race-schedule \
  --query-races-per-step 10 \
  --learning-rate 0.00003 \
  --early-stopping-patience 4 \
  --fine-tune-scope icl_and_race_head \
  --min-race-number 3 \
  --race-context-mode self_attention \
  --seed 42 \
  --device cpu \
  --classification_loss_weight 1.0 \
  --pairwise_loss_weight 0.25 \
  --attention_delta_pairwise_loss_weight 0.05 \
  --cardinality_loss_weight 0.0


"open_price",
    "fluc1",
    "fluc2",
    

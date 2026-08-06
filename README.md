python train_model.py \
  --resume-model /home/theo/yy1/outputs/tabfm_race_top3.pt \
  --output /home/theo/yy1/outputs/tabfm_race_top3.pt \
  --fine-tune-scope full_model \
  --epochs 50 \
  --steps-per-epoch 50 \
  --context-races-per-step 4 \
  --query-races-per-step 50 \
  --learning-rate 2e-6 \
  --early-stopping-patience 8 \
  --min-race-number 6 \
  --race-context-mode self_attention \
  --pairwise-loss-weight 1.0 \
  --attention-delta-pairwise-loss-weight 0.5 \
  --cardinality-loss-weight 0 \
  --stress-top3-recall-max-drop 0.02 \
  --seed 42 \
  --device cpu \
  --zero-features




  python train_model.py \
    --resume-model /home/theo/yy1/outputs/tabfm_race_top3_lr3e5_1.pt \
    --output /home/theo/yy1/outputs/tabfm_race_top3_lr3e5_1.pt \
    --epochs 20 \
    --query-races-per-step 1 \
    --context-races-per-step 96 \
    --learning-rate 0.00003 \
    --early-stopping-patience 2 \
    --min-race-number 5 \
    --race-context-mode self_attention \
    --seed 42 \
    --device cpu \
    --classification_loss_weight 1.0 \
    --pairwise_loss_weight 0.25 \
    --attention_delta_pairwise_loss_weight 0.05 \
    --cardinality_loss_weight 0.0
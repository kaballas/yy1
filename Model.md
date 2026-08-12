# Race-Aware TabFM for Top-3 Horse Racing Prediction


## Overview

This repository contains a race-aware extension of **TabFM**, a transformer-based model for tabular prediction. The model is adapted to horse racing by combining:

1. **Tabular feature embeddings**
2. **Column-wise and row-wise transformer processing**
3. **In-context learning (ICL) from historical labelled races**
4. **Race-level self-attention across runners in the same field**
5. **A historical top-3 / non-top-3 prototype branch**
6. **Label-aware cross-attention from query runners to historical runners**
7. **Race-aware ranking losses**
8. **Chronological context sampling that matches inference-time use**

The prediction target is **`top3_mask`**:

- `1` = the runner finished in the top three
- `0` = the runner did not finish in the top three

The model therefore outputs a **top-3 probability for each runner**, not a win probability.

For a race with runners \(r_1,\dots,r_n\), the model estimates

\[
P(y_i = 1 \mid x_i,\; \text{field},\; \text{historical context}),
\]

where \(y_i=1\) means that runner \(i\) belongs to the actual top three.

---

# 1. Motivation

Horse-racing prediction is not purely an independent tabular classification problem.

A runner's chance of finishing in the top three depends on:

- its own characteristics;
- the quality and composition of the current field;
- how similar runners performed historically;
- the relative strength of rivals;
- chronological information available before the race.

A conventional tabular classifier often evaluates each runner independently:

\[
P(y_i=1)=f(x_i).
\]

The proposed race-aware TabFM instead learns approximately:

\[
P(y_i=1)
=
f(
x_i,
\{x_j:j\in R_i\},
\mathcal{C}_i
),
\]

where:

- \(x_i\) is the feature vector for runner \(i\);
- \(R_i\) is the set of runners in the same race;
- \(\mathcal{C}_i\) is a set of strictly earlier historical races used as in-context examples.

This architecture allows the model to reason about both **runner quality** and **runner quality relative to the field**.

---

# 2. Model Inputs

The core model consumes tensors with the following logical structure.

## 2.1 Feature tensor `x`

```python
x.shape == [batch_size, sequence_rows, feature_count]
```

Each row represents one runner.

For example, a runner could contain features such as:

```text
speed_rating
barrier
weight
recent_form
jockey_rating
distance_rating
market_features
track_features
...
```

The exact ordered feature list is defined by the feature manifest supplied with:

```bash
--features-json
```

Before entering the model, numerical data is median-imputed and standardized:

\[
x' = \frac{x-\mathrm{median}}{\mathrm{scale}}.
\]

The preprocessing statistics are fitted on training data and stored with the checkpoint.

---

## 2.2 Target tensor `y`

```python
y.shape == [batch_size, sequence_rows]
```

The binary target is:

```text
0 = not top 3
1 = top 3
```

During training, historical and query rows have their known target values available to the training loop, but `train_size` determines which rows are treated as context by the model.

During inference, query labels are unknown and use the sentinel:

```python
-100
```

---

## 2.3 `train_size`

```python
train_size.shape == [batch_size]
```

For each sequence, `train_size[b]` is the number of historical context rows.

If a sequence contains 80 historical runner rows followed by an 8-runner target race:

```text
rows 0-79   = historical context
rows 80-87  = query race
```

then:

```python
train_size[b] = 80
```

Thus:

\[
\text{context row} \iff t < \text{train\_size},
\]

and

\[
\text{query row} \iff t \ge \text{train\_size}.
\]

---

## 2.4 `race_group_ids`

```python
race_group_ids.shape == [batch_size, sequence_rows]
```

This tensor identifies which runners belong to the same race.

Example:

```text
Historical Race 100:
100 100 100 100 100 100 100 100

Historical Race 101:
101 101 101 101 101 101 101 101

Query Race 125:
125 125 125 125 125 125 125 125
```

The actual integer value is only an identifier. Equality means that rows belong to the same race.

Padding rows use:

```python
race_group_id = -1
```

---

## 2.5 `valid_row_mask`

Because independently contextualized races can contain different numbers of rows, batch sequences are padded.

```python
valid_row_mask.shape == [batch_size, sequence_rows]
```

The mask is:

```text
True  = real runner row
False = padded row
```

This ensures padded rows do not contribute to race attention, losses, or predictions.

---

# 3. Chronological Training Layout

A key design decision is that each query race receives its **own historical context**.

With:

```bash
--query-races-per-step 5
```

one optimizer step contains approximately:

```text
Batch item 0:
historical races before Query Race A
+
Query Race A

Batch item 1:
historical races before Query Race B
+
Query Race B

Batch item 2:
historical races before Query Race C
+
Query Race C

Batch item 3:
historical races before Query Race D
+
Query Race D

Batch item 4:
historical races before Query Race E
+
Query Race E
```

The five query races do **not** share a single random context.

For every query race, the sampler selects the most recent eligible races that are:

1. strictly earlier in time;
2. from the same competition;
3. complete, with at least four runners and exactly three positive `top3_mask` labels.

Optimizer queries draw context from the eligible optimizer pool. Validation
queries may also use earlier completed races from the validation partition.
This is causal: an earlier validation outcome may become context for a later
validation race, but the current or any future race cannot.

This is important because it makes training resemble deployment.

No future race is allowed to become context for an earlier race.

---

# 4. Overall Architecture

The full architecture can be summarized as:

```text
Raw runner features
        |
        v
CellEmbedder
        |
        v
Column Set Transformer
        |
        v
Row Transformer
        |
        v
Column Set Transformer #2
        |
        v
Row Transformer #2
        |
        v
Runner representation
        |
        v
RaceSetEncoder
(pre-ICL field interaction)
        |
        v
In-Context Learning Transformer
        |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
Base logits              Prototype branch      Label-aware historical
        |                                       cross-attention
        v                                              |
RaceSetHead                                             |
(post-ICL race attention)                               |
        |                                               |
        +----------------------+------------------------+
                   |
                   v
             Final logits
                   |
                   v
             Binary softmax
                   |
                   v
             P(finish top 3)
```

The current implementation can enable all four race/context extensions:

```bash
--race-context-mode self_attention
--encode-races-before-icl
--context-prototype-branch
--label-context-branch
--label-context-labels-in-values-only
```

---

# 5. Cell Embedding

Raw scalar features are first transformed into learned embeddings.

For numerical features, the model uses learned Fourier-style transformations based on sine and cosine functions.

Conceptually:

\[
z(x)=W[\sin(xF),\cos(xF)],
\]

where:

- \(x\) is a feature value or feature group;
- \(F\) contains learned frequencies;
- \(W\) projects the expanded representation into the model embedding dimension.

Categorical and numerical paths have separate learned frequency/projection parameters.

This produces a vector representation for each table cell.

---

# 6. Column Processing

The first column stage uses a **Set Transformer** with induced self-attention.

For a given feature, the model can process values across the historical/query sequence while using an attention mask so that the context structure is respected.

This allows a value such as a standardized speed rating to be interpreted relative to examples available in the current context rather than only through a fixed global transformation.

The Set Transformer uses learned inducing vectors, which reduce the cost of attending over large tabular sequences.

---

# 7. Row Interaction

After column processing, learned classification tokens are added and a row transformer processes each runner across its feature representations.

For one runner, the model can therefore combine information such as:

```text
speed
+
barrier
+
weight
+
recent form
+
distance suitability
+
jockey / trainer information
+
other features
```

into a compact runner-level representation.

The model applies:

```text
Column interaction
→ Row interaction
→ Column interaction
→ Row interaction
```

before producing the final per-runner representation used by the ICL stage.

---

# 8. Pre-ICL RaceSetEncoder

The option:

```bash
--encode-races-before-icl
```

activates a race-level transformer **before** in-context learning.

All runners belonging to one race are packed together:

```text
Runner A ----\
Runner B -----\
Runner C ------> Race self-attention
Runner D -----/
Runner E ----/
```

For each runner representation \(h_i\), the encoder learns a correction:

\[
\Delta h_i
=
g(h_i,\{h_j:j\in R_i\}),
\]

and returns:

\[
h_i' = h_i + \Delta h_i.
\]

This residual structure means the race branch modifies the existing TabFM representation rather than replacing it.

The output projection is initialized at zero, so the race-aware branch begins close to the base model behavior and can learn corrections during training.

Because historical races are also grouped when pre-ICL race encoding is enabled, the model can represent a historical runner relative to the field it actually faced.

---

# 9. In-Context Learning

The ICL module is one of the central components.

Historical context rows contain both:

```text
runner representation
+
known top-3 label
```

The model embeds the historical label and adds it only to context rows.

For classification:

\[
r_t =
h_t
+
\mathbf{1}[t < N_c]\,E_y(y_t),
\]

where:

- \(h_t\) is the runner representation;
- \(N_c\) is `train_size`;
- \(E_y\) is the learned target-label embedding.

The transformer then uses the labelled historical examples as in-context information for the query runners.

Conceptually, the model can learn:

```text
Historical runner with this field-relative profile -> TOP3
Historical runner with another profile             -> NOT TOP3
Today's runner has a similar representation        -> increase TOP3 score
```

The query labels themselves are not injected.

---

# 10. Post-ICL RaceSetHead

With:

```bash
--race-context-mode self_attention
```

a second field-level attention mechanism operates after the ICL transformer.

The ICL module first produces:

```text
base logits
+
hidden runner representation
```

The RaceSetHead then groups runners from the same query race and applies transformer self-attention across the field.

It produces a class-logit correction:

\[
\Delta z_i^{race}.
\]

Final logits contain this race-aware correction:

\[
z_i'
=
z_i^{base}
+
\Delta z_i^{race}.
\]

This allows the model to reconsider a runner after observing the complete current field.

For example, a strong runner may receive a different correction depending on whether the race contains weak or exceptionally strong rivals.

---

# 11. Context Prototype Branch

The model also uses:

```bash
--context-prototype-branch
--context-prototype-dim 16
--context-prototype-max-correction 0.25
```

The prototype head creates two learned historical prototypes:

\[
p_0 = \text{mean representation of historical non-top3 runners},
\]

\[
p_1 = \text{mean representation of historical top3 runners}.
\]

Runner and prototype vectors are normalized in a learned 16-dimensional metric space.

For a query runner representation \(q_i\), the head compares:

\[
s_0 = q_i^\top p_0,
\]

\[
s_1 = q_i^\top p_1.
\]

The useful direction is:

\[
s_i = s_1-s_0.
\]

This is converted into a bounded correction:

\[
c_i =
c_{max}\tanh(g\,s_i),
\]

where:

```text
c_max = 0.25
```

in the illustrated configuration. The value is configurable with
`--context-prototype-max-correction`.

The two-class correction is:

\[
[-c_i,\; +c_i].
\]

Thus a runner that resembles historical top-3 examples more than historical non-top-3 examples receives a positive class-1 correction.

The branch is deliberately bounded so it supplements rather than dominates the main model.

## 11.1 Label-Aware Historical Cross-Attention

With:

```bash
--label-context-branch
--label-context-heads 2
--label-context-max-correction 0.25
--label-context-labels-in-values-only
```

the learned runner representations before ICL provide a second, explicit
historical-label pathway. A query runner representation is used as an attention
query. Each strictly earlier historical runner representation is used as an
attention key. Its representation augmented by a learned embedding of its
binary `top3_mask` label is used as the corresponding value:

\[
k_j = \operatorname{Norm}(r_j),
\]

\[
v_j = \operatorname{Norm}(r_j + E_y(y_j)),
\]

\[
h_i^{context} = \operatorname{MHA}(q=r_i,\;k=k_j,\;v=v_j).
\]

Consequently, changing historical labels changes the retrieved values and the
resulting correction, but it does not change which historical runners receive
attention for fixed runner representations.

The attended result is normalized, projected to two logits, bounded with
`tanh`, and emitted only for query rows:

\[
\Delta z_i^{label\_context}
=
c_{max}\tanh(W h_i^{context}).
\]

Query labels are never injected into this branch. Padded rows and historical
rows receive zero correction. During cached inference, prefill stores detached
historical representations, labels, and their validity mask; decode attends to
that same memory.

---

# 12. Final Prediction

For every query runner, the model produces two logits:

```text
z0 = not top 3
z1 = top 3
```

The inference code computes:

\[
P(\mathrm{top3})
=
\frac{e^{z_1}}
{e^{z_0}+e^{z_1}}.
\]

In PyTorch:

```python
query_top3_probability = torch.softmax(query_logits, dim=-1)[:, 1]
```

These probabilities are marginal runner probabilities.

They are **not a softmax across runners in a race**.

Therefore, for an 8-runner race:

```text
Runner A  0.78
Runner B  0.69
Runner C  0.62
Runner D  0.48
Runner E  0.33
Runner F  0.27
Runner G  0.18
Runner H  0.14
```

the predicted top-three set is obtained by ranking the runners and taking the three highest scores.

---

# 13. Training Objective

The objective is configurable. A full race-aware example is:

```bash
--classification-loss-weight 1.0
--pairwise-loss-weight 0.25
--attention-delta-pairwise-loss-weight 0.05
--context-prototype-loss-weight 0.25
--label-context-loss-weight 0.25
--cardinality-loss-weight 0.0
--context-dependence-loss-weight 0.1
--context-dependence-margin 0.005
```

The effective objective is:

\[
\mathcal{L}
=
1.0\,\mathcal{L}_{cls}
+
0.25\,\mathcal{L}_{pair}
+
0.05\,\mathcal{L}_{race\_delta}
+
0.25\,\mathcal{L}_{prototype}
+
0.25\,\mathcal{L}_{label\_context}
+
0.10\,\mathcal{L}_{context}
+
0.0\,\mathcal{L}_{card}.
\]

Each race-based component is constructed so that complete races contribute approximately equally rather than allowing large fields to dominate solely because they contain more runners. Disabled branches contribute zero and their direct loss weights must also be zero.

---

# 14. Classification Loss

For each query runner, weighted two-class cross entropy is calculated.

The per-runner losses are first averaged inside each race, and the race losses are then averaged.

This gives the main supervised signal:

```text
actual top-3 runner     -> class 1
actual non-top-3 runner -> class 0
```

For the example above, the classification-loss weight is:

```text
1.0
```

---

# 15. Pairwise Race Ranking Loss

Classification alone does not guarantee a useful within-race ordering.

The model therefore defines the runner ranking score:

\[
s_i = z_{i,1}-z_{i,0}.
\]

Within each query race, every true top-3 runner is compared against every non-top-3 runner.

If \(p\) is positive and \(n\) is negative:

\[
\mathcal{L}_{pair}(p,n)
=
\operatorname{softplus}(-(s_p-s_n)).
\]

The loss becomes small when:

\[
s_p > s_n.
\]

For an 8-runner race with exactly 3 top-3 runners and 5 non-top-3 runners, there are:

\[
3\times5=15
\]

positive-negative comparisons.

This directly trains the model to rank the real top three above the remainder of the field.

Weight:

```text
0.25
```

---

# 16. Race-Attention Delta Ranking Loss

The RaceSetHead correction is also trained directly.

Instead of applying the pairwise objective only to final logits, the training loop extracts the race-attention delta and applies the same positive-vs-negative ranking criterion.

This teaches the race-attention branch itself to make useful field-relative adjustments:

\[
\Delta s_{top3}
>
\Delta s_{non-top3}.
\]

Weight:

```text
0.05
```

This is intentionally smaller than the main classification/ranking objectives.

---

# 17. Prototype Pairwise Loss

The prototype branch receives its own direct pairwise ranking supervision.

The prototype correction is trained so that actual top-3 runners receive a more favorable prototype-based score than non-top-3 runners in the same race.

Weight:

```text
0.25
```

This prevents the prototype branch from becoming an unused auxiliary pathway.

## 17.1 Label-Context Pairwise Loss

When label-aware historical cross-attention is enabled, its correction receives
the same direct within-race positive-versus-negative supervision:

```bash
--label-context-loss-weight 0.25
```

This requires the retrieval branch itself to rank true top-three query runners
above non-top-three runners instead of allowing it to learn only uniform
confidence shifts. The final classification, pairwise, and context-dependence
objectives still supervise the combined model output.

The preferred retrieval mode separates attention addressing from outcome
content:

```text
query runner representation -> query
historical runner representation -> key
historical runner representation + historical label embedding -> value
```

Enable it with `--label-context-labels-in-values-only`. Runner similarity
therefore selects the historical evidence, while the known historical outcome
changes only the information retrieved from that evidence. Legacy checkpoints
that injected labels into both keys and values preserve that behavior until
explicitly upgraded and retrained.

---

# 18. Context-Dependence Loss

A possible failure mode of an ICL architecture is that the model learns to ignore the historical labels.

The training code explicitly tests against this.

For every enabled context-dependence step:

1. run the model with correct historical context labels;
2. randomly permute only the historical context labels;
3. run the model again;
4. require the correct historical labels to produce a better prediction loss.

Let:

\[
L_{correct}
\]

be the normal prediction loss and

\[
L_{perm}
\]

the loss after historical-label permutation.

The context-dependence loss is:

\[
\mathcal{L}_{context}
=
\max(
0,
\operatorname{stopgrad}(L_{correct})
+
m
-
L_{perm}
),
\]

where:

\[
m=0.005.
\]

The correct-context loss is detached in this auxiliary term so gradients do not simply cancel between the correct and permuted runs.

This objective encourages the network to make genuine use of historical outcomes.

Weight:

```text
0.1
```

---

# 19. Cardinality Loss

The code also supports a top-three cardinality objective.

If \(p_i=P(y_i=1)\), the cardinality loss for one race is:

\[
\mathcal{L}_{card}
=
\left(
\frac{\sum_i p_i - 3}{3}
\right)^2.
\]

This encourages the sum of marginal top-three probabilities in a complete race to approach three.

For example, a run using:

```bash
--cardinality-loss-weight 0.0
```

disables this constraint.

When its weight is zero, runner probabilities are **not explicitly encouraged to
sum to 3**. A positive weight adds the penalty during optimization, but it does
not impose a hard constraint and does not guarantee an exact sum at inference.

This distinction should be stated clearly when discussing probability calibration.

Before changing this objective, run `evaluate_model_stages.py` on genuinely
held-out races. It reports the mean, median, p10, p90, and mean absolute error
of the probability sum at the base, label-context, race-head, and final stages,
plus its relationship with top-three recall. Do not repair the sum by inference
normalisation: these are independent Bernoulli marginals, so such a transform
would change their meaning without supplying a valid constrained model.

## 19.1 Post-ICL race-head residual scale

For self-attention race context, the additive path is:

\[
z_{after\ race}=z_{base}+s_{race}\,\Delta z_{race}.
\]

`--race-head-scale` controls \(s_{race}\). The default `1.0` preserves old
checkpoint behaviour, and old trusted module checkpoints receive `1.0` while
loading. The raw and scaled residuals are both retained in debugger/training
auxiliary output so branch supervision can still inspect the raw head. Scale
selection must use held-out stage ablations, not one illustrative race.

---

# 20. Automatic Race Schedule

The training run used:

```bash
--auto-race-schedule
--query-races-per-step 5
```

In automatic mode, the requested number of query races acts as a target maximum.

If there are \(N\) eligible training races:

\[
\text{query races per step}
=
\min(5,N),
\]

and:

\[
\text{steps per epoch}
=
\left\lceil
\frac{N}{\text{query races per step}}
\right\rceil.
\]

The schedule is built so each eligible race appears as a query at least once per epoch, with only small repetition if \(N\) is not divisible by the query batch size.

Thus, in automatic mode an epoch represents systematic exposure to the eligible
race set rather than an arbitrary fixed number of mini-batches. With automatic
mode disabled, `--steps-per-epoch` is used exactly as supplied.

---

# 21. Example: One 8-Runner Query Race

Suppose the query race is:

```text
Aster
Bolt
Comet
Dancer
Echo
Falcon
Grit
Halo
```

and the true finish labels are:

```text
Aster   1
Bolt    0
Comet   0
Dancer  1
Echo    0
Falcon  1
Grit    0
Halo    0
```

The model receives:

```text
most recent earlier same-competition races
+
the 8 query runners
```

The historical rows form the context prefix.

The query race forms the suffix.

During training:

```text
train_size = number of historical runner rows
```

and the top-three labels of the query runners are used only to calculate the loss.

The model may produce:

```text
Aster   0.78
Falcon  0.73
Dancer  0.66
Halo    0.46
Bolt    0.38
Echo    0.27
Comet   0.18
Grit    0.13
```

The predicted top-three set is then:

```text
Aster
Falcon
Dancer
```

The training metrics compare this predicted ranking against the three actual positives.

---

# 22. Evaluation Metrics

The training system uses whole-race ranking metrics, including:

## Top-3 recall

Fraction of the three actual top-3 runners recovered among the model's top-three ranked runners.

For one race:

```text
actual top 3    = A, C, F
predicted top 3 = A, F, D
```

then:

\[
\text{Top3 Recall}=\frac{2}{3}.
\]

---

## Exact top-3 set rate

A race is counted as correct only when the predicted top-three set exactly matches the actual top-three set, regardless of internal order.

---

## Contained-in-top-k metrics

The implementation also measures whether all three actual top-3 runners occur somewhere in the model's:

```text
top 4
top 5
top 6
```

These metrics are useful because they quantify near-miss ranking quality.

---

## ROC AUC and log loss

The model also computes binary runner-level probability metrics.

ROC AUC evaluates ranking separation between top-3 and non-top-3 runners.

Log loss evaluates the quality/calibration of runner-level top-three probabilities.

---

# 23. Inference

Native SQLite inference follows the causal validation-context contract.

For every target race:

1. identify races strictly earlier than the target;
2. require `status='finished'`, complete binary labels, and at least four runners;
3. restrict context to the target's `competition_id`;
4. select the most recent checkpoint-sized context window;
5. concatenate historical runner features with the target race;
6. assign historical top-3 labels;
7. assign query labels to `-100`;
8. build race group IDs;
9. run the model;
10. extract the target-race logits;
11. apply two-class softmax;
12. rank runners by \(P(\mathrm{top3})\).

The native loader reads earlier completed context from `race_runners`, not only
from `tabfm_trainable_validation_runners`. This is necessary for competitions
reserved entirely for validation and matches training-time validation, where an
earlier completed validation race can causally contextualize a later one.

Generic CSV/Parquet prediction instead uses the labelled historical rows present
in the supplied file and can optionally enforce chronology with `--date-column`.

This matching of training and inference context structure is an important part of the experimental design.

---

# 24. Reference Training Commands

## 24.1 Base-model collapse diagnostic

When the auxiliary race, prototype, and label-context branches are disabled, a
continued base-model run can use:

```bash
python train_model.py \
  --resume-model outputs/base_smoke.pt \
  --output outputs/base_smoke1.pt \
  --training-csv output/base_smoke_training.csv \
  --validation-csv output/base_smoke_validation.csv \
  --features-json tabfm_features.json \
  --epochs 20 \
  --steps-per-epoch 100 \
  --query-races-per-step 10 \
  --probe-races 10 \
  --probe-every-steps 10 \
  --context-races-per-step 9 \
  --learning-rate 0.0001 \
  --fine-tune-scope full_model \
  --seed 42 \
  --device cpu \
  --classification-loss-weight 1.0 \
  --pairwise-loss-weight 0.25 \
  --cardinality-loss-weight 1.0 \
  --checkpoint-metric loss \
  --early-stopping-patience 10 \
  --max-grad-norm 1 \
  --allow-high-fine-tune-learning-rate \
  --print-race-schedule
```

The explicit feature manifest must exactly match the resumed checkpoint's saved
feature order. Changing features requires a fresh output with
`--overwrite-existing`, not `--resume-model`.

`--query-races-per-step` is a batch-size control, not a dataset-size limit, and
`--probe-races` limits only the deterministic diagnostic probe. Validation is
never truncated by `--max-valid-races`; that argument is a compatibility check.

---

# 25. Interpretation of the Trained System

A useful summary is:

> The model predicts whether each runner will finish in the top three by combining the runner's own tabular characteristics with labelled examples from recent historical races. Optional field-relative self-attention, label-aware retrieval, and top-3/non-top-3 prototype branches can add further corrections when enabled.

More formally, the final runner score can be interpreted conceptually as:

\[
z_i^{final}
=
z_i^{ICL}
+
\Delta z_i^{race}
+
\Delta z_i^{label\_context}
+
\Delta z_i^{prototype}.
\]

The ICL term learns from historical labelled examples.

The race term adjusts the runner relative to the current field.

The prototype term directly measures similarity to historical positive and negative classes.

The label-context term directly retrieves relevant historical runner/outcome
associations using learned representations.

---

# 26. Important Methodological Properties

## Whole-race grouping

Runners in the same race are processed together for race attention. A race should not be split across separate race-conditioned prediction calls.

## Chronological causality

Only strictly earlier races are used as historical context.

## Same-competition context

Historical context is restricted to the query race's competition. Native
inference fails clearly when fewer than the checkpoint-required number of
eligible earlier races exist.

## Equal-per-race loss aggregation

Race-level loss aggregation reduces bias toward races with larger fields.

## Explicit ranking supervision

The model is trained not only to classify runners but also to rank top-3 runners above non-top-3 runners.

## Context-use regularization

Historical labels are deliberately permuted in an auxiliary counterfactual pass to ensure that correct context is useful.

---

# 27. Limitations

Several limitations should be acknowledged in a paper.

### Marginal rather than joint top-three probabilities

Each runner receives a binary top-three probability. The model does not directly define a joint probability distribution over all valid three-runner finishing sets.

### Cardinality is a soft objective

The cardinality term is optional. Even with a positive weight, it encourages but
does not enforce a sum of three. Calibration and per-race probability sums must
therefore be measured on held-out chronological races.

### Ranking does not model finishing order

The target is membership in the top three, not exact finishing place. First, second, and third are all positive examples.

### Context sensitivity

Because historical labels and historical fields influence prediction, results can depend on which chronological context races are available.

### Complete-race requirement

Race-conditioned attention requires the full relevant field to be provided together.

### Data leakage must be controlled externally

Features themselves must contain only information available before race start. Chronological context prevents future-label leakage, but it cannot protect against a feature that already encodes future information.

---

# 28. Suggested Paper Description

A concise methods description suitable for adaptation into a paper is:

> We developed a race-aware extension of TabFM for binary top-three classification. Each runner was represented as a row of standardized tabular features. TabFM cell, column, and row transformations first produced runner-level representations. We introduced a pre-ICL race-set encoder that applied permutation-equivariant self-attention among runners belonging to the same historical or query race. The resulting field-aware representations were processed by the model's in-context learning transformer, which used labelled runners from the most recent strictly earlier same-competition races as contextual examples. A label-aware retrieval branch additionally used each query representation to attend to keys formed from historical runner representations and values formed from those representations plus learned outcome-label embeddings. A second race-set attention head generated field-relative corrections to query logits, while an auxiliary prototype branch compared query runners against learned historical top-three and non-top-three prototypes. Training combined equal-per-race weighted classification loss, within-race positive-versus-negative pairwise ranking loss, direct pairwise supervision of each correction branch, and a context-dependence margin objective based on permutation of historical context labels. Predictions were obtained from the class-1 softmax probability and ranked within each race to identify the predicted top-three runners.

---

# 29. Reproducibility Notes

For reproducible reporting, record at minimum:

- feature manifest and feature order;
- preprocessing medians and scales;
- random seed;
- chronological train/validation split;
- context races per prediction;
- query races per step;
- learning rate;
- number of epochs and effective steps per epoch;
- model architecture arguments;
- all auxiliary loss weights;
- checkpoint-selection rule;
- exact software and PyTorch versions;
- CPU/GPU execution environment.

For the base-model continuation example in Section 24, the key settings are:

```text
seed = 42
learning rate = 1e-4
device = CPU
query races per step = 10
steps per epoch = 100
epochs = 20
race context = disabled
pre-ICL race encoder = disabled
prototype branch = disabled
label-context branch = disabled
cardinality loss weight = 1.0
checkpoint metric = validation loss
```

---

# 30. Summary

The race-aware TabFM model combines five forms of reasoning:

```text
1. Feature reasoning
   What characteristics does this runner have?

2. Field reasoning
   How strong is this runner relative to today's rivals?

3. Historical in-context reasoning
   How did comparable runners and fields perform in recent races?

4. Prototype reasoning
   Does this runner resemble historical top-3 runners
   more than historical non-top-3 runners?

5. Label-aware retrieval
   Which historical runner representations are relevant to this query,
   and what labelled outcomes were attached to them?
```

The resulting model is therefore not simply an independent binary classifier.

It is a **chronological, field-aware, in-context tabular transformer trained jointly for top-three classification and within-race ranking**.

---

# 31. RaceFormerTop3: Current-Race-Only Alternative

`RaceFormerTop3` is a separate model family for experiments that must not use
historical labelled examples. It consumes exactly one current race at a time:

```text
runner features -> runner MLP -> optional race transformer
                -> optional [RACE] summary -> one logit per runner -> sigmoid
```

It has no `train_size`, historical context prefix, ICL decoder, context labels,
prototype branch, label embedding, or same-competition history requirement.
Padding permits several complete races in one optimizer batch, but attention
never crosses between batch items.

The three controlled variants are:

```text
independent  Model A: runner MLP only
transformer  Model B: runner MLP + current-field self-attention
race_token   Model C: Model B + learned [RACE] summary token
```

All variants use one sigmoid logit per runner and an equal-per-race objective:

\[
\mathcal{L}=\mathcal{L}_{BCE}
+\lambda_{rank}\mathcal{L}_{rank}
+\lambda_{card}\mathcal{L}_{card}.
\]

Defaults are `1.0`, `0.5`, and `0.1` respectively. Cardinality remains a soft
penalty, not a constrained normalization.

Train Model C:

```bash
python train_raceformer.py \
  --variant race_token \
  --features-json tabfm_features.json \
  --output outputs/raceformer_c.pt \
  --epochs 30 \
  --races-per-batch 32 \
  --model-dim 128 \
  --attention-heads 4 \
  --race-layers 2 \
  --dropout 0.10 \
  --learning-rate 0.0003 \
  --ranking-loss-weight 0.50 \
  --cardinality-loss-weight 0.10 \
  --device cpu
```

Change only `--variant` to run the A/B/C architecture comparison. For a small
pipeline test, `--max-training-races 10 --max-validation-races 10` selects the
most recent ten eligible races from each partition. Unlike TabFM context
training, all ten selected training races can be optimizer queries because this
model requires no preceding context window.

Predict one race:

```bash
python predict_raceformer.py \
  --checkpoint outputs/raceformer_c.pt \
  --race-id 10785832 \
  --device cpu
```

Backtest the same chronological validation partition and compare with market:

```bash
python predict_raceformer.py \
  --checkpoint outputs/raceformer_c.pt \
  --backtest \
  --backtest-max-races 20 \
  --device cpu
```

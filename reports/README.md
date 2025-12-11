# Reports

This folder contains **PDF run reports** produced by the training/evaluation pipeline for a small set of “reference experiments”. Each PDF is a self-contained snapshot of:

- The **exact configuration** used (dataset, model, training, evaluation)
- The **headline metrics** (train/val, test accuracy, rolling accuracy, oracle rate, constraint checks)
- A standard set of **plots**:
  - training/validation curves
  - rolling accuracy on a single long exploration
  - rolling accuracy across **graph-boundary switches**
  - example train/test graphs
  - action-id histograms

These reports are meant to answer two questions quickly:

1. Does the model solve the “easy” setting where train/test share semantics?
2. Does it **still** solve the “hard” setting where train/test triplets are disjoint (no overlap), forcing **online adaptation**?

## How to read a report (quick checklist)

Open any PDF and scan in this order:

1. **Run Configuration (page 1)**
   - `dataset type`: `standard` vs `disjoint`
   - `num nodes`, `num actions`, `num timesteps`
   - `network type`: `rnn` vs `plastic`
   - (plastic only) `plastic batch size`, and the implied fast-weight footprint

2. **Key Metrics (page 2)**
   - `triplet overlap fraction`
     - `100%` means standard (train/test share (u, a, v) patterns)
     - `0%` means disjoint (strict holdout of triplets)
   - `test next node accuracy`
     - “one-step” accuracy over the test set
   - `avg rolling accuracy single graph` / `avg rolling accuracy graph set`
     - accuracy measured over long rollouts (after the model has had time to *experience* the graph)

3. **Rolling accuracy plots (pages 4–5)**
   - **Single graph**: does performance ramp up quickly and stay high?
   - **Graph boundaries**: do you see sharp drops at boundaries followed by fast recovery?

## What runs are included

### RNN baseline (no fast weights)

- `rnn_8nodes_10actions_standard.pdf`
  - Small graph, standard dataset (train/test share triplets).
  - This is the “easy” sanity check for the baseline.

- `rnn_8nodes_10actions_disjoint.pdf`
  - Same size, but disjoint dataset (no triplet overlap).
  - This is the key failure case: a model that relies on global memorization should collapse here.

### Plastic network (fast-weight lookup table)

- `plastic_8nodes_10actions_standard.pdf`
  - Same small setting as the RNN sanity check, but with the plastic mechanism enabled.

- `plastic_8nodes_10actions_disjoint.pdf`
  - The canonical “online adaptation” test on a small environment.
  - Expectation: strong performance despite zero triplet overlap.

- `plastic_50nodes_15actions_standard.pdf`
  - Larger graph/action space stress test in the standard setting.

- `plastic_50nodes_15actions_disjoint.pdf`
  - Larger stress test in the disjoint setting.
  - Expectation: the model should still adapt quickly in long rollouts, even when one-step accuracy is harsher.

## Headline outcomes (what these reports demonstrate)

Across these reports you should see the intended qualitative story:

- **RNN**
  - Works in `standard`.
  - Breaks in `disjoint` (low test accuracy; poor rolling accuracy).

- **Plastic**
  - Works in `standard`.
  - Continues to work in `disjoint`, showing fast recovery after graph switches in the boundary plots.

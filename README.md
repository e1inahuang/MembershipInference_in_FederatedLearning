# Membership Inference Attack on Federated Learning (NIST PPFL Genomics)

Black-box Membership Inference Attack (MIA) against CNN target models trained on the **NIST PPFL Soybean Genomics** dataset under three privacy regimes (No-DP / DP ε=10 / DP ε=200). Implemented in PyTorch with IBM's [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox).

> **Course context:** CMU — Engineering Privacy (final project). Group members: Elina Yutong Huang, Isabel Agadagba, Cole Callahan.

---

## Background

The project originally targeted the **Pima Indians Diabetes** dataset with a TensorFlow Federated (TFF) pipeline and ART-based MIA. Mid-project we hit unresolvable version conflicts between `tensorflow-federated ≤ 2.17`, `numpy ≤ 1.25`, and `adversarial-robustness-toolbox ≥ 2.13`. To unblock the attack experiments, this branch **pivots to the NIST PPFL reference setup** — pre-trained CNN target models on soybean genomic records — which is fully PyTorch and plays cleanly with ART. The TFF/Pima branch lives in the companion repo [e1inahuang/MembershipInference_in_FederatedLearning](https://github.com/e1inahuang/MembershipInference_in_FederatedLearning); this repo is the ART/CNN branch documented in the notebook.

Reference setup: <https://github.com/rivera-lanasm/nist_ppfl/blob/master/NIST_PPFL_problem1_202503/submission/README.md>

---

## Task

Given a federated learning system with **4 clients** each holding a disjoint shard of soybean genomic data (125,766 features per record), determine — for a set of *challenge records* — whether each record was part of some client's training set, and if so, **which client**. The attacker has only black-box access to each client's trained model (logits on chosen inputs).

---

## Models Evaluated

Three pre-trained target models per client are loaded from `.torch` checkpoints. Architectures live in `attack_targets/{cnn,dpcnn10,dpcnn200}/model.py`:

| Privacy type | Directory          | DP budget (ε) | Description                |
|--------------|--------------------|:-------------:|----------------------------|
| `cnn`        | `attack_targets/cnn/`        | — (no DP)     | Baseline 1-D CNN           |
| `dpcnn10`    | `attack_targets/dpcnn10/`    | 10            | Differentially-private CNN |
| `dpcnn200`   | `attack_targets/dpcnn200/`   | 200           | Differentially-private CNN |

Attack model: `art.attacks.inference.membership_inference.MembershipInferenceBlackBox`, instantiated per client/target via `utils.build_attack_model(task_model, num_data_features, hyperparameters_path)`.

---

## Data Layout

Per client the loader expects three record files plus the trained model and its hyperparameters:

```
attack_targets/
├── cnn/
│   ├── cnn_challenge_records.dat
│   └── client_<i>/
│       ├── cnn_<i>.torch
│       ├── cnn_<i>_relevant_records.dat     # members
│       ├── cnn_<i>_external_records.dat     # non-members
│       └── cnn_<i>_hyperparameters.json
├── dpcnn10/                                  # same layout, filenames use `dpcnn_<i>` prefix
└── dpcnn200/                                 # same layout, filenames use `dpcnn_<i>` prefix
```

Record counts (relevant / external) — Client 1: 73/64, Client 2: 95/83, Client 3: 59/52, Client 4: 23/20. Client 4 also ships `cnn_4_challenge_members.json` (ground truth for the challenge set, used for validation).

---

## Pipeline

1. **Load** target model, relevant records, external records, hyperparameters via `utils.load_model` / `utils.load_data` / `utils.load_path_set`.
2. **Wrap** the target in an ART `PyTorchClassifier`.
3. **Train attack model** with `MembershipInferenceBlackBox.fit(x=rel_x, y=rel_y, test_x=ext_x, test_y=ext_y, pred=rel_preds, test_pred=ext_preds)` — members = relevant set, non-members = external set. Features: target-model logits + raw (x, y).
4. **Score challenge records** against all 4 client attack models; take argmax probability across clients, threshold at **0.5** — records below the threshold are assigned class `0` ("no client").
5. **Emit submission** as `{privacy_type}_submission_file.csv` with columns `index, prediction`.

---

## Experiments

### Part 1 — Compare the three privacy regimes

`compare_models()` runs `evaluate_membership_inference(pt)` for `pt ∈ {cnn, dpcnn10, dpcnn200}`, across all 4 clients, and writes:

- `{privacy_type}_evaluation_results.png` — per-client confidence histograms + confusion matrices
- `privacy_model_comparison.csv` — accuracy, positive/negative precision & recall, F1
- `{privacy_type}_submission_file.csv` — challenge predictions

### Part 2 — Hyperparameter tuning against DPCNN10, Client 3

`optimize_parameters_client3()` sweeps the attack's `batch_size ∈ {16, 32, 64, 128}` and `epochs ∈ {5, 10, 20, 30}`. Objective is framed from the **defender's** side: **lower attack F1 is better**. Outputs:

- `client3_parameter_optimization.png`
- `client3_best_confusion_matrix.png`
- `optimized_hyperparameters.json`

Threshold sweep helper `analyze_threshold_effect(pt, thresholds=[0.3..0.8])` produces `{privacy_type}_threshold_analysis.{png,csv}`.

---

## Metrics

Every run reports: accuracy, member (positive) precision/recall, non-member (negative) precision/recall, F1, full confusion matrix, and member-vs-non-member probability histograms.

---

## Requirements

- Python 3.10+, PyTorch, torchvision, torchinfo
- `adversarial-robustness-toolbox`
- numpy, pandas, matplotlib, seaborn

Notebook was developed on Google Colab; dataset + `attack_targets/` tree must be mounted at the working directory before the evaluation cells are run.

---

## Usage

Open `MembershipInferenceFL.ipynb` and run top-to-bottom. Main entry points:

```python
# Part 1: all three privacy types, all four clients
compare_models()

# Part 2: tune attack hyperparameters on DPCNN10 / client 3
optimize_parameters_client3()

# Single privacy type
results = evaluate_membership_inference("cnn")   # or "dpcnn10", "dpcnn200"
visualize_results(results, "cnn")
```

---

## Repo Contents

| File | Purpose |
|------|---------|
| `MembershipInferenceFL.ipynb` | Full attack + experiments pipeline (primary artifact) |
| `Federated Learning Final Report.docx` | Final report — documents the earlier TFF/Pima branch and the pivot |
````

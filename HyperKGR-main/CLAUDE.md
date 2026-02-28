# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HyperKGR (Hyperbolic Knowledge Graph Reasoning) performs knowledge graph reasoning in hyperbolic space using GNNs that encode symbolic paths. Published at EMNLP 2025.

## Dependencies

- PyTorch (1.9.1+ / 1.12.1)
- torch_scatter 2.0.9
- numpy, scipy
- CUDA GPU required

## Running Commands

### Transductive reasoning

```bash
cd sample/transductive    # or not_sample/transductive
python3 train.py --data_path ./data/WN18RR/ --train --topk 1000 --layers 8 --fact_ratio 0.96 --gpu 0
```

### Inductive reasoning

```bash
cd sample/inductive       # or not_sample/inductive
python3 train.py --data_path ./data/WN18RR_v1
```

### Evaluation with saved checkpoint

```bash
python3 train.py --data_path ./data/WN18RR/ --eval --topk 1000 --layers 8 --gpu 0 --weight ./data/WN18RR/8-layers-best.pt
```

## Architecture

### Two Variants: `sample/` vs `not_sample/`

- **`sample/`**: Includes adaptive node sampling (top-k selection with Gumbel-Softmax during training, regular softmax during eval). Each GNN layer prunes the propagation frontier to `topk` nodes per batch via `W_samp` scoring + straight-through estimator. The `sample/` variant has learnable curvature (`nn.Parameter`).
- **`not_sample/`**: No node sampling; all reachable neighbors propagate. Uses fixed curvature. Simpler baseline.

Both variants have `transductive/` and `inductive/` sub-directories with parallel file structures.

### Core Files (per variant)

| File | Role |
|------|------|
| `train.py` | Entry point. Contains dataset-specific hyperparameter configs hardcoded per dataset name. Parses args, sets up model, runs training/eval loop. |
| `models.py` | `GNNLayer` and `GNNModel`. The core hyperbolic GNN. Contains all Poincare ball math (expmap0, logmap0, mobius_add, project, hyp_distance). |
| `base_model.py` | `BaseModel` wraps `GNNModel` with optimizer, scheduler, train_batch(), evaluate(), save/load. |
| `load_data.py` | `DataLoader` reads KG triples, builds sparse adjacency (`csr_matrix`), provides `get_neighbors()` for layer-wise subgraph extraction. |
| `utils.py` | Ranking metrics: `cal_ranks()` (filtered ranking), `cal_performance()` (MRR, Hits@1, Hits@10). |

### GNN Message Passing Pipeline

Each `GNNLayer.forward()`:
1. Compute attention: `alpha = sigmoid(W(ReLU(Ws*hs + Wr*hr + Wqr*h_qr)))`
2. Map to hyperbolic: `expmap0(hs, c)`, `expmap0(hr, c)`
3. Hyperbolic translation: `project(mobius_add(hs, hr, c), c)` (TransE-style in Poincare ball)
4. Map back to tangent: `logmap0(message, c)`
5. Scatter-aggregate: `scatter(alpha * message, obj, reduce='sum')`
6. Output transform: `W_h -> expmap0 -> act -> logmap0`
7. GRU gate combines current hidden with previous layer's hidden state

### Key Differences: Transductive vs Inductive

- **Transductive**: Single entity set. `shuffle_train()` re-splits facts/train each epoch using `fact_ratio`. Graph uses separate `KG`/`tKG` for train/test.
- **Inductive**: Separate entity sets for train (`entity2id`) and test (`entity2id_ind`). Train graph (`tra_KG`) and inductive graph (`ind_KG`) are fully disjoint. Validation uses transductive test set; test uses inductive valid+test.

### Data Format

Datasets need: `entities.txt`, `relations.txt`, `facts.txt`, `train.txt`, `valid.txt`, `test.txt`. For transductive, facts/train are split from original train.txt at ratio controlled by `--fact_ratio`. For inductive, a parallel `_ind/` directory holds the inductive graph.

### Evaluation Metrics

Filtered ranking protocol: MRR, Hits@1, Hits@10. Results logged to `results/{dataset}/` text files.

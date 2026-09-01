# Structure-Guided Neighbor Discovery (SG-GNN)

Code for:

- **"Adapting to Heterophilic Graph Data with Structure-Guided Neighbor Discovery"**, Victor M. Tenorio, Madeline Navarro, Samuel Rey, Santiago Segarra, and Antonio G. Marques. Submitted to *IEEE Transactions on Knowledge and Data Engineering* (under review).
- **"Structure-Guided Input Graph for GNNs facing Heterophily"**, the preceding conference version by the same authors, presented at the Asilomar Conference on Signals, Systems, and Computers, 2024.

SG-GNN addresses heterophily in graph-based node classification not by changing the GNN's
propagation rule, but by *constructing alternative input graphs* from structural node
attributes (role-based and global centrality features), and adaptively combining a GNN's
predictions across several such graphs.

## Installation

```bash
pip install -r requirements.txt
```

Requires PyTorch and PyTorch Geometric (see [pyg.org](https://pytorch-geometric.readthedocs.io)
for CUDA-specific install instructions). All datasets (Texas/Wisconsin/Cornell, Actor,
Chameleon/Squirrel, Cora/CiteSeer, USA/Brazil/Europe, Roman-Empire) are downloaded
automatically by PyTorch Geometric on first use (cached under `~/.datapyg`); no manual data
download is needed.

The `alternatives/` baseline wrappers (GRAND, CD-GNN, LG-GNN) and `arch.py`'s other baselines
need no extra dependencies. `ablations/embeddings.py`'s `DeepWalk`/`Node2Vec`/`Struc2Vec`
graphs additionally need [GraphEmbedding](https://github.com/shenweichen/GraphEmbedding), and
its `GraphWave` graphs need a GraphWave implementation exposing
`graphwave.graphwave.graphwave_alg` (Donnat et al., "Learning Structural Node Embeddings via
Diffusion Wavelets", KDD 2018) -- both optional, only required for `precompute_candidate_graphs.py`.

## Repository structure

```
arch.py           GNN architectures (SG-GNN and baselines) -- see its module docstring for
                   how class names here map to the paper's notation (SG-GCN, SG-GCN_N, SG-GCN_L, ...)
train.py           Generic train/eval loop with early stopping, shared by every experiment script
utils.py           Structural attribute computation (role-based, global), data loading, masks

exp_main_matched.py              Main comparison table (Table VI, 11 datasets x 8 baselines x 6 SG-GNN variants)
exp_newbaselines_matched.py      Adds the GRAND / CD-GNN / LG-GNN rows to the same table
exp_largegraph_matched.py        Roman-Empire column of Table VI (large-graph accuracy)
exp_largegraph_timing.py         Roman-Empire time/memory, fixed hidden dim (Table XIII)
exp_largegraph_timing_matched.py Roman-Empire time/memory, parameter-fair capacity (supplementary; see response to reviewers)

alternatives/      Self-contained reimplementations of the GRAND, CD-GNN, and LG-GNN baselines
                    (not copies of the original authors' code -- see each wrapper's docstring)

ablations/
  feature_importance.py          Leave-one-group-out feature ablation (Table VII)
  sensitivity_sweep.py           k / epsilon-ball / R sensitivity sweeps (Tables VIII, IX, X)
  embeddings.py                  Feature/embedding-graph classes used by the R sweep's candidate pool
  precompute_candidate_graphs.py Precomputes that candidate pool (run before sensitivity_sweep.py's R sweep)
  bound_tightness.py             Empirical tightness of the error bound, Theorem 1 (Table XI)
  timing_memory.py               Per-epoch time/memory on the original 11 datasets (Table XII)

notebooks/
  Experiments_conference.ipynb   Conference-paper (Asilomar 2024) experiments and figures; kept
                                  for reference, uses `dgl` rather than the PyTorch Geometric
                                  pipeline above -- see the notebook's first cell
```

Every script under the repository root and `ablations/` is run directly (`python <script>.py`,
or `python precompute_candidate_graphs.py` followed by `python sensitivity_sweep.py` from inside
`ablations/`); configuration (hyperparameters, datasets, output paths) is set at the top of each
file rather than via command-line flags, following this project's experiment convention. Each
script's module docstring documents its exact output files and which paper table/figure they
produce. Outputs are written to `results/<experiment_name>/` (git-ignored) as `.npy`/`.pkl`/`.json`,
never committed to this repository.

## Reproducing the paper

**Tables VI (main comparison, 11 datasets + Roman-Empire):**
```bash
python exp_main_matched.py            # 8 baselines + 6 SG-GNN variants, 11 datasets
python exp_newbaselines_matched.py    # + GRAND, CD-GNN, LG-GNN rows
python exp_largegraph_matched.py      # + Roman-Empire column
```

**Table VII (feature-group ablation):** `python ablations/feature_importance.py`

**Tables VIII-X (k / epsilon-ball / R sensitivity):**
```bash
cd ablations
python precompute_candidate_graphs.py   # only needed once, for the R sweep
python sensitivity_sweep.py
```

**Table XI (bound tightness):** `python ablations/bound_tightness.py`

**Table XII (small-graph timing):** `python ablations/timing_memory.py`

**Table XIII (Roman-Empire timing):** `python exp_largegraph_timing.py` (and, for the
parameter-fair supplementary comparison discussed in the response to reviewers,
`python exp_largegraph_timing_matched.py`)

Some hyperparameters and implementation details that are stated only briefly in the paper for
space (e.g., the exact per-baseline capacity used for parameter fairness, or the training
schedule -- Adam, learning rate, weight decay, early-stopping patience) are fully specified at
the top of the corresponding script above.

The results underlying Tables I-V (the original submission's graph-construction and
graph-quality results) and the paper's figures were produced during earlier development and, for
the conference-paper subset, are reproduced in `notebooks/Experiments_conference.ipynb`; `utils.py`'s
structural-attribute computation is the same code used throughout, including for those tables.

## Citing

```bibtex
@inproceedings{tenorio2024structure,
    title={Structure-Guided Input Graph for GNNs facing Heterophily},
    author={Victor M. Tenorio and Madeline Navarro and Samuel Rey and Santiago Segarra and Antonio G. Marques},
    year={2024},
    booktitle={Asilomar Conference on Signals, Systems, and Computers}
}
```

A citation for the journal version will be added here once it is accepted / assigned a DOI.

# Structure-Guided Neighbor Discovery (SG-GNN)

Code for **"Adapting to Heterophilic Graph Data with Structure-Guided Neighbor Discovery"**,
Victor M. Tenorio, Madeline Navarro, Samuel Rey, Santiago Segarra, and Antonio G. Marques.
Submitted to *IEEE Transactions on Knowledge and Data Engineering* (under review). (A preceding
conference version, "Structure-Guided Input Graph for GNNs facing Heterophily", appeared at
Asilomar SSC 2024; see `notebooks/Experiments_conference.ipynb`.)

SG-GNN addresses heterophily in graph-based node classification not by changing the GNN's
propagation rule, but by *constructing alternative input graphs* from structural node
attributes (role-based and global centrality features), and adaptively combining a GNN's
predictions across several such graphs.

## Installation

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric are **not** in `requirements.txt` -- install them separately,
matching your CUDA version (see
[pytorch-geometric.readthedocs.io/.../installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html));
`requirements.txt` records the exact versions this code was run with. All datasets
(Texas/Wisconsin/Cornell, Actor, Chameleon/Squirrel, Cora/CiteSeer, USA/Brazil/Europe,
Roman-Empire) are downloaded automatically by PyTorch Geometric on first use (cached under
`~/.datapyg`); no manual data download is needed.

`sggnn.baselines` (GRAND, CD-GNN, LG-GNN) and every other model in `sggnn.models` need no extra
dependencies. `sggnn.embeddings`' `DeepWalk`/`Node2Vec`/`Struc2Vec` graphs additionally need
[GraphEmbedding](https://github.com/shenweichen/GraphEmbedding) installed separately, and its
`GraphWave` graphs need a GraphWave implementation exposing `graphwave.graphwave.graphwave_alg`
(Donnat et al., "Learning Structural Node Embeddings via Diffusion Wavelets", KDD 2018) --
neither is vendored or patched in this repository. Both are optional and only needed for
`python main.py build-graphs`; Role/Global/Feat graphs need nothing beyond `requirements.txt`.

### Graph cache

Every accuracy table except VIII, IX and XIII reads a precomputed cache of candidate graphs,
`node_embeddings/embedding_graphs.npz` (~34 MB). Build it with:

```bash
python main.py build-graphs
```

(needs the optional embedding dependencies above for the DeepWalk/Node2Vec/Struc2Vec/GraphWave
columns; takes on the order of hours, dominated by Struc2Vec/GraphWave on Actor and Squirrel).
Regenerated graphs are algorithmically equivalent to the ones used for the paper, but for
embedding methods with non-deterministic training (the random walks behind DeepWalk/Node2Vec/
Struc2Vec, GraphWave's heat-kernel sampling) **not bit-identical on every dataset** -- expect
Table IV/VI/VII/X numbers involving those columns to differ from the paper by noise-level
amounts. The Role/Global/Feat columns, and everything derived only from them, are deterministic
and exactly reproducible.

Do **not** upload or commit `embedding_graphs.npz`; distribute it as a GitHub Release asset if
you want to skip rebuilding it (`GRAPH_CACHE=/path/to/embedding_graphs.npz python main.py <target>`
points any target at an existing cache instead of the default `node_embeddings/embedding_graphs.npz`).

## Repository structure

```
main.py            Single entry point, `python main.py <target>` -- see the table below
sggnn/             Library
  models.py          GNN architectures (SG-GNN and baselines) -- see its docstring for how
                     class names map to the paper's notation (SG-GCN, SG-GCN_N, SG-GCN_L, ...).
                     FAGCN follows the official model (bdy9527/FAGCN): input projection to
                     hidden_dim, FAConv stack at hidden_dim width (eps=0.3), no activation
                     between FAConv layers; capacity is matched via hidden_dim like every
                     other baseline. The activation after the input projection is the
                     project-wide `nonlin` (Tanh), not the official repo's ReLU.
  train.py           Generic train/eval loop with early stopping
  data.py            Dataset loading, train/val/test splits (random or public -- see
                     get_data_dict's `split` argument, default 'random'), the graph-cache
                     reader, parameter-matching helpers (count_params, find_matched_hid/...)
  features.py        Role-based/global structural attributes, k-NN and epsilon-ball graph
                     builders (dense and sparse, the latter for Roman-Empire)
  embeddings.py      DeepWalk/Node2Vec/Struc2Vec/GraphWave graph builders + build_cache()
                     (the compute_embeddings.py logic, called by `main.py build-graphs`)
  baselines/         Self-contained reimplementations of GRAND, CD-GNN, LG-GNN (not copies
                     of the original authors' code -- see each module's docstring)
  paths.py           results/<RESULTS_DIR>/<experiment_id>/ output-path convention, plus
                     the git-commit and JSON helpers every experiment uses
  report.py          Small helpers shared by the analysis/ scripts

experiments/       One module per paper result (raw numbers only, no LaTeX)
  graph_eval.py      Table IV (graph/embedding comparison) + Fig. 4's Adaptive-GNN/SG-GNN rows
  graph_timing.py    Table V (graph-construction timing)
  main_table.py      Table VI (merges the former exp_main_matched/exp_newbaselines_matched/
                     exp_largegraph_matched/make_table6_merge; MAIN_TABLE_PART={main,
                     newbaselines,largegraph,merge} selects the part, default 'main')
  coefficients.py    Fig. 4 (learned adaptive coefficients alpha_r)
  feature_importance.py   Table VII (leave-one-group-out feature ablation)
  sensitivity.py     Tables VIII/IX/X (k / epsilon-ball / R sensitivity; merges the former
                     sensitivity_sweep.py + sensitivity_sweep_v2.py; SENSITIVITY_PARTS=k,eps,r).
                     k and eps sweeps use random splits by default (SENS_SPLIT=public for each
                     dataset's own masks instead); R sweep always uses random splits and
                     reproduces its view subsets deterministically (zlib.crc32).
  bound_synth.py     Section VI-E, synthetic CSBM error-bound tightness (no numbered table)
  bound_real.py      Section VI-E, real-dataset error-bound tightness
  timing.py          Tables XII/XIII (end-to-end training time; merges timing_total.py +
                     timing_total_largegraph.py; TIMING_PARTS=table12,table13)

analysis/          Read saved artifacts only, build LaTeX/figures -- never retrains anything
  table4.py, table5.py, table6.py, sensitivity_tables.py (VIII-X), timing_table.py (XII)
  plot_bound_synth.py, plot_bound_real.py, plot_feature_importance.py, plot_timing_largegraph.py

node_embeddings/   Empty except for a .gitkeep; embedding_graphs.npz goes here (git-ignored)

notebooks/
  Intuitions.ipynb              Tables I-III / Figs. 1-2 (early graph-construction intuitions)
  Experiments_conference.ipynb  Conference-paper (Asilomar 2024) experiments, uses `dgl`
```

Every experiment/analysis module is runnable directly (`python experiments/<x>.py`,
`python analysis/<x>.py`) or through `main.py`; configuration (hyperparameters, datasets,
device, output directory) is set at the top of each file and overridable via environment
variables documented in its own docstring (e.g. `MATCHED_DEVICE`, `MATCHED_NSIMS`, `SENS_*`,
`RESULTS_DIR`, `GRAPH_CACHE`), following this project's experiment convention rather than
command-line flags. Outputs go to `results/<RESULTS_DIR>/<experiment_id>/results.{pkl,json}`
(default `RESULTS_DIR=./results`, git-ignored), never committed to this repository.

## Reproducing the paper

| Paper result | Command |
|---|---|
| Table IV (graph/embedding comparison) | `python main.py table4` |
| Table V (graph-construction timing) | `python main.py table5` |
| Table VI (main comparison, 11 datasets + Roman-Empire) | `python main.py table6` |
| Fig. 4 (learned adaptive coefficients) | `python main.py fig4` |
| Table VII (feature-group ablation) | `python main.py table7` |
| Table VIII (k sensitivity) | `python main.py table8` |
| Table IX (epsilon-ball sensitivity) | `python main.py table9` |
| Table X (R / number-of-views sensitivity) | `python main.py table10` |
| Section VI-E, real-dataset bound experiment | `python main.py bound-real` |
| Section VI-E, bound-tightness table/summary | `python main.py table11` (after `bound-real`) |
| Table XII (small-graph timing) | `python main.py table12` |
| Table XIII (Roman-Empire timing) | `python main.py table13` |
| Candidate-graph cache | `python main.py build-graphs` |
| Section VI-E, synthetic CSBM bound | `python experiments/bound_synth.py` (no numbered table; not wired into `main.py`) |
| Tables I-III, Figs. 1-2 | `notebooks/Intuitions.ipynb` |

`python main.py --help` lists every target with a one-line note on whether it needs the graph
cache; `python main.py <target> --quick` runs a fast, reduced-settings smoke test for the
targets cheap enough for that (table4, fig4, table5, table7-10).

Some hyperparameters and implementation details stated only briefly in the paper for space
(e.g., the exact per-baseline capacity used for parameter fairness, or the training schedule --
Adam, learning rate, weight decay, early-stopping patience) are fully specified at the top of
the corresponding `experiments/` module.

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

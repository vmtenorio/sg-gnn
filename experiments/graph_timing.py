"""
Table V (`tab:comp_times_embs`): wall-clock time to build each candidate
graph type (Feat, Role, Global, DeepWalk, Node2Vec, Struc2Vec, GraphWave).
Times the full `EmbeddingGraph` construction (attribute/embedding computation
+ k-NN graph + epsilon-ball graph, nneigh=3 default) once per
(dataset, attribute type), matching the methodology used for the published
numbers. Needs the optional embedding dependencies (see README) for the
DeepWalk/Node2Vec/Struc2Vec/GraphWave columns; RoleFeat/GlobalFeat/Feat need
nothing beyond this package.

Output: results/<RESULTS_DIR>/table5/results.{pkl,json}
"""
import os
import sys
import time
import json
import pickle
import logging
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from sggnn import embeddings
from sggnn.data import load_dataset
from sggnn.paths import results_dir

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

EXPERIMENT_ID = 'table5'
DATASETS = os.environ.get('TABLE5_DATASETS', ','.join(embeddings.DATASETS)).split(',')
EMBEDDING_NAMES = os.environ.get('TABLE5_EMBEDDINGS', ','.join(embeddings.EMBEDDING_NAMES)).split(',')

OUT_DIR = results_dir(EXPERIMENT_ID)
comp_times = np.zeros((len(DATASETS), len(EMBEDDING_NAMES)))

for d, dname in enumerate(DATASETS):
    logging.info(f"Dataset {d+1}/{len(DATASETS)}: {dname}")
    data = load_dataset(dname)
    for e, embname in enumerate(EMBEDDING_NAMES):
        t_start = time.time()
        emb_model_cls = getattr(embeddings, embname + 'Embeddings')
        emb_model_cls(data[0], verbose=False)
        comp_times[d, e] = time.time() - t_start
        logging.info(f"  {embname}: {comp_times[d, e]:.3f} s")

payload = {
    'datasets': DATASETS, 'embedding_names': EMBEDDING_NAMES, 'comp_times': comp_times,
    'metadata': {'experiment_id': EXPERIMENT_ID, 'timestamp': datetime.datetime.now().isoformat()},
}
with open(OUT_DIR / 'results.pkl', 'wb') as f:
    pickle.dump(payload, f)
with open(OUT_DIR / 'results.json', 'w') as f:
    json.dump({**payload, 'comp_times': comp_times.tolist()}, f, indent=2)

logging.info(f"Done. Saved to {OUT_DIR}")

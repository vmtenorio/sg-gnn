"""Shared output-path convention: results/<experiment_id>/{results.pkl,
results.json, ...}, rooted at a configurable directory (env var RESULTS_DIR,
default "./results"), per experiment i.e. per module in experiments/; plus the
helpers every experiment uses to write its metadata and JSON artifacts."""
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def results_dir(experiment_id: str) -> Path:
    base = Path(os.environ.get('RESULTS_DIR', './results'))
    out = base / experiment_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def git_commit(short=False):
    """HEAD commit hash of this repository, or None outside a git checkout."""
    cmd = ['git', 'rev-parse'] + (['--short'] if short else []) + ['HEAD']
    try:
        return subprocess.check_output(cmd, cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def to_json_safe(obj):
    """Convert arrays, tensors and numpy scalars (recursively) to JSON types."""
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    return obj


def nan_to_none(obj):
    """Replace float NaNs in nested dicts/lists with None (strict-JSON null)."""
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    return obj

# humaidclf/stratify.py
from __future__ import annotations
import math, random
import pandas as pd
from collections import defaultdict
from typing import List

def stratified_k_shards(
    df: pd.DataFrame,
    label_col: str,
    k: int,
    seed: int = 42,
    shuffle_within_class: bool = True,
) -> List[pd.DataFrame]:
    """
    Split df into k shards while preserving class ratios as closely as possible.

    Rules:
    - We partition each class's rows across shards (round-robin) after an optional shuffle.
    - Works even when minority classes have < k rows: some shards will get 0 of that class.
    - Keeps approximate global ratios per shard.

    Returns: list of k dataframes (may differ in size by at most 1 per class chunk).
    """
    assert k >= 2, "k must be >= 2"
    rng = random.Random(seed)

    # Bucket rows by class
    buckets = {}
    for cls, g in df.groupby(label_col, dropna=False):
        g = g.copy()
        if shuffle_within_class:
            # stable shuffle with seed
            idx = list(g.index)
            rng.shuffle(idx)
            g = g.loc[idx]
        buckets[cls] = list(g.index)

    # Prepare k index lists
    shard_idx = [[] for _ in range(k)]

    # Round-robin distribute per-class rows across shards
    for cls, idxs in buckets.items():
        for j, idx in enumerate(idxs):
            shard_idx[j % k].append(idx)

    # Build shard dataframes, preserve original row order within each shard
    shards = []
    for j in range(k):
        part = df.loc[sorted(shard_idx[j])].copy()
        shards.append(part.reset_index(drop=True))
    return shards

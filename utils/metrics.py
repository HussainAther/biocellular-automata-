# utils/metrics.py

import numpy as np
from scipy.stats import entropy

def binary_entropy(grid):
    """
    Calculates the Shannon entropy of a binary grid.
    """
    values, counts = np.unique(grid, return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs, base=2)

def temporal_entropy(history):
    """
    Returns entropy per frame, useful for detecting chaotic vs. stable behavior.
    """
    return [binary_entropy(frame) for frame in history]

def symmetry_score(grid):
    """
    Returns a basic horizontal symmetry score.
    1.0 = perfect symmetry, 0.0 = no symmetry.
    """
    flipped = np.fliplr(grid)
    similarity = np.sum(grid == flipped)
    total = grid.size
    return similarity / total

def detect_static_end(history):
    """
    Checks if the CA has reached a static state.
    """
    return np.all(history[-1] == history[-2])

def activity_score(history):
    """
    Measures how many cells change on average per step.
    """
    diffs = [np.sum(history[i] != history[i-1]) for i in range(1, len(history))]
    return np.mean(diffs)


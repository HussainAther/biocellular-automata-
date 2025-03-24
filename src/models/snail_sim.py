# snail_sim.py

import numpy as np

def snail_shell_rule(grid, neighbor_count, num_states):
    """
    Simulates shell pigmentation pattern formation.
    Basic idea: a moving window that 'paints' depending on local state.
    
    This is a 1D CA unrolled as a 2D grid:
    - Rows = time
    - Columns = shell circumference
    """
    new_grid = np.zeros_like(grid)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            left = grid[i-1, j-1] if j > 0 else 0
            center = grid[i-1, j]
            right = grid[i-1, j+1] if j < grid.shape[1]-1 else 0
            total = left + center + right
            if total == 1:
                new_grid[i, j] = 1
            else:
                new_grid[i, j] = 0
    return new_grid


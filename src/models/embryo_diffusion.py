# embryo_diffusion.py

import numpy as np
from scipy.ndimage import convolve

def embryo_diffusion_rule(grid, neighbor_count, num_states):
    """
    Simulates a reaction-diffusion-style CA to model embryo patterning.
    Inspired by Turing patterns (but discrete).
    
    Multi-state CA with activator/inhibitor logic.
    States: 0 = off, 1 = activator, 2 = inhibitor
    """
    new_grid = np.zeros_like(grid)
    activator_mask = (grid == 1)
    inhibitor_mask = (grid == 2)

    # Diffusion kernels for each chemical
    kernel = np.array([[0.05, 0.2, 0.05],
                       [0.2,  -1,  0.2],
                       [0.05, 0.2, 0.05]])

    act_field = convolve(activator_mask.astype(float), kernel, mode='wrap')
    inh_field = convolve(inhibitor_mask.astype(float), kernel, mode='wrap')

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if act_field[i, j] > inh_field[i, j]:
                new_grid[i, j] = 1  # Activator wins
            elif inh_field[i, j] > 0.3:
                new_grid[i, j] = 2  # Inhibitor dominates
            else:
                new_grid[i, j] = 0  # Stay off

    return new_grid


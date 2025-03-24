# utils/visualize.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def plot_grid(grid, title="CA Grid", cmap="binary"):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap=cmap, interpolation='nearest')
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_history(history, title="CA Evolution", cmap="binary"):
    """
    history: 3D array of shape (time, height, width)
    For 1D CAs unrolled over time, shape might be (time, width)
    """
    if history.ndim == 3:
        # 2D CA
        fig, axes = plt.subplots(1, history.shape[0], figsize=(history.shape[0], 4))
        for i, ax in enumerate(axes):
            ax.imshow(history[i], cmap=cmap)
            ax.axis("off")
        plt.suptitle(title)
        plt.show()
    elif history.ndim == 2:
        # 1D CA unrolled
        plt.figure(figsize=(8, 6))
        plt.imshow(history, cmap=cmap, interpolation='nearest', aspect='auto')
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

def animate_history(history, interval=100, save_path=None, cmap="binary"):
    """
    Create an animation of CA evolution.
    """
    fig = plt.figure()
    im = plt.imshow(history[0], cmap=cmap, animated=True)

    def update(i):
        im.set_array(history[i])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=interval, blit=True)

    if save_path:
        ani.save(save_path, fps=1000 // interval)
    else:
        plt.show()

    return ani


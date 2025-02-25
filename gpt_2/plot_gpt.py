### Scatter plot of local Lipschitz constant of inputs with respect to the maximal radius of their tokens and the number of tokens.

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("gpt_2/figures").mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)

layers = torch.load("gpt_2/datasets/layers.torch")

plt.rcParams["font.family"] = "Times New Roman"

### Alice in Wonderland

print("Alice in Wonderland")

ns = torch.load("gpt_2/datasets/alice/ns.torch")

coef = [35.3, 25.0]
pow = [0.26, 0.51]
labels = [r"$c n^{0.26}$", r"$c n^{0.51}$"]

for k, b in enumerate(layers):

    to_plot = torch.load(f"gpt_2/datasets/alice/to_plot_{b}.torch")

    ## radius plot
    f, ax = plt.subplots(figsize=(3, 2))
    for i, n in enumerate(ns):
        radius, lipschitz_csts = to_plot[i]
        color = plt.cm.viridis(n / 100.0)
        ax.scatter(radius, lipschitz_csts, s=2, alpha=0.7, color=color)
    x_axis = np.linspace(0.0, max(radius))
    ax.set_xlabel("Mean Radius")
    ax.set_ylabel("Lipschitz Constant")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"gpt_2/figures/fig_radius_alice_{b}.pdf")

    ## n plot
    max_radius_list = []
    min_radius_list = []
    for i, n in enumerate(ns):
        radius, _ = to_plot[i]
        max_radius_list.append(max(radius))
        min_radius_list.append(min(radius))
    max_radius = max(max_radius_list)
    min_radius = min(min_radius_list)
    f, ax = plt.subplots(figsize=(2.5, 2))
    for i, n in enumerate(ns):
        radius, lipschitz_csts = to_plot[i]
        color = plt.cm.plasma((radius - min_radius) / (max_radius - min_radius))
        ax.scatter(
            n * torch.ones(len(lipschitz_csts)), lipschitz_csts, s=2, color=color
        )
    x_axis = np.linspace(0.0, 100.0)
    ax.plot(
        x_axis, x_axis ** pow[k] * coef[k], c="darkblue", linewidth=0.9, label=labels[k]
    )
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Lipschitz Constant")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"gpt_2/figures/fig_n_alice_{b}.pdf")

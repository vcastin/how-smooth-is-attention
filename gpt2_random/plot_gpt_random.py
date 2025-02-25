### Scatter plot of local Lipschitz constant of inputs with respect to the maximal radius of their tokens and the number of tokens.

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("gpt2_random/figures").mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)

layers = torch.load("gpt2_random/datasets/layers.torch")

plt.rcParams["font.family"] = "Times New Roman"

### Alice in Wonderland

print("Alice in Wonderland")

ns = torch.load("gpt2_random/datasets/alice/ns.torch")


for k, b in enumerate(layers):

    to_plot = torch.load(f"gpt2_random/datasets/alice/to_plot_{b}.torch")

    ## radius plot
    f, ax = plt.subplots(figsize=(2.5, 2))
    for i, n in enumerate(ns):
        radius, lipschitz_csts = to_plot[i]
        color = plt.cm.viridis(n / 100.0)
        ax.scatter(radius, lipschitz_csts, s=1.5, alpha=0.7, color=color)
    x_axis = np.linspace(0.0, max(radius))
    ax.set_xlabel("Mean Radius")
    ax.set_ylabel("Lipschitz Constant")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"gpt2_random/figures/fig_radius_alice_{b}.pdf")

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
        threshold = 0.1
        radius = radius[lipschitz_csts > threshold]
        lipschitz_csts = lipschitz_csts[lipschitz_csts > threshold]
        color = plt.cm.autumn((radius - min_radius) / (max_radius - min_radius))
        ax.scatter(
            n * torch.ones(len(lipschitz_csts)), lipschitz_csts, s=1.5, color=color
        )
    x_axis = np.linspace(0.0, 100.0)
    if b == 6:
        ax.plot(
            x_axis,
            x_axis**0.25 * 0.445,
            label=r"$cn^{1/4}$",
            c="darkblue",
            linewidth=0.9,
        )
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Lipschitz Constant")
    ax.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"gpt2_random/figures/fig_n_alice_{b}.pdf")

### AG_NEWS

print("AG_NEWS")

ns = torch.load("gpt2_random/datasets/AG_NEWS/ns.torch")

for k, b in enumerate(layers):

    to_plot = torch.load(f"gpt2_random/datasets/AG_NEWS/to_plot_{b}.torch")

    ## radius plot
    f, ax = plt.subplots(figsize=(2.5, 2))
    for i, n in enumerate(ns):
        radius, lipschitz_csts = to_plot[i]
        lipschitz_csts = lipschitz_csts**0.5
        threshold = 0.1
        radius = radius[lipschitz_csts > threshold]
        lipschitz_csts = lipschitz_csts[lipschitz_csts > threshold]
        color = plt.cm.viridis(n / 100.0)
        ax.scatter(radius, lipschitz_csts, s=1.5, alpha=0.7, color=color)
    x_axis = np.linspace(0.0, max(radius))
    ax.set_xlabel("Mean Radius", loc="left")
    ax.set_ylabel("Lipschitz Constant")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"gpt2_random/figures/fig_radius_NEWS_{b}.pdf")

    ## n plot
    max_radius_list = []
    min_radius_list = []
    for i, n in enumerate(ns):
        radius, lipschitz_csts = to_plot[i]
        lipschitz_csts = lipschitz_csts**0.5
        threshold = 0.1
        radius = radius[lipschitz_csts > threshold]
        lipschitz_csts = lipschitz_csts[lipschitz_csts > threshold]
        max_radius_list.append(max(radius))
        min_radius_list.append(min(radius))
    max_radius = max(max_radius_list)
    min_radius = min(min_radius_list)
    f, ax = plt.subplots(figsize=(2.5, 2))
    for i, n in enumerate(ns):
        radius, lipschitz_csts = to_plot[i]
        threshold = 0.1
        radius = radius[lipschitz_csts > threshold]
        lipschitz_csts = lipschitz_csts[lipschitz_csts > threshold]
        color = plt.cm.autumn((radius - min_radius) / (max_radius - min_radius))
        ax.scatter(
            n * torch.ones(len(lipschitz_csts)), lipschitz_csts, s=1.5, color=color
        )
    x_axis = np.linspace(0.0, 100.0)
    if b == 6:
        ax.plot(
            x_axis,
            x_axis**0.25 * 0.445,
            label=r"$cn^{1/4}$",
            c="darkblue",
            linewidth=0.9,
        )
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Lipschitz Constant")
    ax.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"gpt2_random/figures/fig_n_NEWS_{b}.pdf")

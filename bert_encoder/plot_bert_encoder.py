### Scatter plot of local Lipschitz constant of inputs with respect to the maximal radius of their tokens and the number of tokens.

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("bert_encoder/figures").mkdir(parents=True, exist_ok=True)

layers = torch.load("bert_encoder/datasets/layers.torch")

plt.rcParams["font.family"] = "Times New Roman"

### Alice in Wonderland

print("Alice in Wonderland")

ns = torch.load("bert_encoder/datasets/alice/ns.torch")


for k, b in enumerate(layers):

    to_plot = torch.load(f"bert_encoder/datasets/alice/to_plot_{b}.torch")

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
    plt.savefig(f"bert_encoder/figures/fig_radius_alice_{b}.pdf")

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
        color = plt.cm.autumn((radius - min_radius) / (max_radius - min_radius))
        ax.scatter(
            n * torch.ones(len(lipschitz_csts)), lipschitz_csts, s=1.5, color=color
        )
    x_axis = np.linspace(0.0, ns[-1])
    if b == 0:
        ax.plot(
            x_axis,
            x_axis**0.24 * 14.3,
            label=r"$cn^{1/4}$",
            c="darkblue",
            linewidth=0.9,
        )
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Lipschitz Constant")
    ax.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"bert_encoder/figures/fig_n_alice_{b}.pdf")

### AG_NEWS

print("AG_NEWS")

ns = torch.load("bert_encoder/datasets/AG_NEWS/ns.torch")

for k, b in enumerate(layers):

    to_plot = torch.load(f"bert_encoder/datasets/AG_NEWS/to_plot_{b}.torch")

    ## radius plot
    f, ax = plt.subplots(figsize=(2.5, 2))
    for i, n in enumerate(ns):
        radius, lipschitz_csts = to_plot[i]
        color = plt.cm.viridis(n / 100.0)
        ax.scatter(radius, lipschitz_csts, s=1.5, alpha=0.7, color=color)
    x_axis = np.linspace(0.0, max(radius))
    ax.set_xlabel("Mean Radius")
    ax.set_ylabel("Lipschitz Constant")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"bert_encoder/figures/fig_radius_NEWS_{b}.pdf")

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
        color = plt.cm.autumn((radius - min_radius) / (max_radius - min_radius))
        ax.scatter(
            n * torch.ones(len(lipschitz_csts)), lipschitz_csts, s=1.5, color=color
        )
    x_axis = np.linspace(0.0, ns[-1])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Lipschitz Constant")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"bert_encoder/figures/fig_n_NEWS_{b}.pdf")

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import BertTokenizer, BertModel, BertConfig, GPT2Config, GPT2Model

plt.rcParams["font.family"] = "Times New Roman"

torch.manual_seed(0)

Path("gpt_2/figures").mkdir(parents=True, exist_ok=True)


def compute_R(input):  # 2-dimensional input
    return np.sqrt(np.mean(np.linalg.norm(input, axis=1) ** 2))


layer = 6
batch_size = 10
to_plot = []
ns = torch.load("gpt_2/datasets/alice/ns.torch")
for n in ns:
    data = torch.load(f"gpt_2/datasets/alice/dataset_{layer}_n={n}.torch").detach()
    for j in range(batch_size):
        R = compute_R(data[j, :, :])
        to_plot.append(R)


f, ax = plt.subplots(figsize=(3, 2.25))
x_axis = np.repeat(ns.numpy(), batch_size)
ax.scatter(x_axis, to_plot, s=2, alpha=0.5)
ax.set_xlabel("Sequence length")
ax.set_ylabel("Average magnitude")
plt.tight_layout()
plt.savefig("gpt_2/figures/correlation_n_R.pdf")

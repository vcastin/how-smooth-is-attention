# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import BertModel, BertConfig, GPT2Config, GPT2Model
from toolbox import get_lipschitz_list

plt.rcParams["font.family"] = "Times New Roman"

torch.manual_seed(0)

Path("adversarial_data/figures").mkdir(parents=True, exist_ok=True)

############## Compute Lipschitz list BERT ##############
# %%

models = [
    BertModel.from_pretrained("bert-base-uncased"),
    BertModel.from_pretrained("bert-base-uncased", is_decoder=True),
    BertModel(BertConfig()),
]

torch.manual_seed(457)
to_plot = []
bound = []
layers = [0, 6]
Rs = [15.5, 21.5]
ns = range(2, 100, 5)
model_name = "bert"
target_heads = [0, 7]
for i in range(2):
    model = models[i]
    layer = layers[i]
    head = target_heads[i]
    R = Rs[i]
    lip_list, bounds = get_lipschitz_list(model, layer, head, model_name, ns, R)
    to_plot.append(lip_list)
# torch.save(torch.Tensor(np.array(to_plot)), "adversarial_data/datasets/for_main_fig_bert.torch")

#################### Plot BERT ####################

csts = [4.39, 5.73]
ns = np.array(ns)
for i, lip_list in enumerate(to_plot):
    f, ax = plt.subplots(figsize=(2.5, 2))
    ax.scatter(ns, lip_list, s=1.5, color="red")  # / ns ** 0.5
    ax.plot(ns, csts[i] * ns**0.5, c="darkblue", label=r"$Cn^{1/2}$", linewidth=0.9)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Lipschitz Constant")
    ax.legend()
    plt.grid()
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"adversarial_data/figures/bert_{i}_adversarial_data.pdf")


############### Compute Lipschitz list GPT ###############
# %%
torch.manual_seed(320)

models = [GPT2Model(GPT2Config()), GPT2Model.from_pretrained("gpt2")]

to_plot = []
bound = []
layer = 6
R = 100
ns = range(2, 100, 5)
model_name = "GPT-2"
head = 0
model = models[0]
lip_list, bounds = get_lipschitz_list(model, layer, head, model_name, ns, R)
to_plot.append(lip_list)
# torch.save(torch.Tensor(np.array(to_plot)), "adversarial_data/datasets/for_main_fig_gpt.torch")

to_plot_5 = []
bound = []
layer = 0
R = 10
ns = range(2, 100, 5)
model_name = "GPT-2"
head = 0
model = models[1]
lip_list, bounds = get_lipschitz_list(model, layer, head, model_name, ns, R)
to_plot_5.append(lip_list)

############### Plot GPT2 #################

cst = 0.92
ns = np.array(ns)
lip_list = np.array(to_plot[0])
f, ax = plt.subplots(figsize=(2.5, 2))
ax.scatter(ns, lip_list, s=1.5, color="red")  # / ns ** 0.5
ax.plot(ns, cst * ns**0.5, c="darkblue", label=r"$Cn^{1/2}$", linewidth=0.9)
ax.set_xlabel("Sequence Length")
ax.set_ylabel("Lipschitz Constant")
ax.legend()
plt.grid()
# plt.show()
plt.tight_layout()
plt.savefig("adversarial_data/figures/gpt_random_adversarial_data.pdf")

cst = 46.9
ns = np.array(ns)
lip_list = np.array(to_plot_5[0])
f, ax = plt.subplots(figsize=(2.5, 2))
ax.scatter(ns, lip_list, s=1.5, color="red")  # / ns ** 0.5
ax.plot(ns, cst * ns**0.5, c="darkblue", label=r"$Cn^{1/2}$", linewidth=0.9)
ax.set_xlabel("Sequence Length")
ax.set_ylabel("Lipschitz Constant")
ax.legend()
plt.grid()
# plt.show()
plt.tight_layout()
plt.savefig("adversarial_data/figures/gpt_pretrained_adversarial_data.pdf")

# %%

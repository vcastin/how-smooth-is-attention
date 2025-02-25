import torch
import numpy as np
import matplotlib.pyplot as plt


def outs(block, input: torch.Tensor):
    # input of shape (n_batch, n_tokens, 768)
    x = block.attention(
        input
    )  # non residual unnormalized attention, 12 attention heads
    return x[0]  # shape = (batch_size, n_tokens, 768)


def loc_lipschitz(attention, Y, threshold=1e-4, max_iter=100, d=768):

    seq_length = Y.shape[1]
    shape = (1, seq_length, d)

    # power iteration
    mus = [0]
    func_vjp = torch.func.vjp(attention, Y)[1]
    i = 0

    u = torch.normal(torch.zeros(shape), torch.ones(shape))
    u /= torch.norm(u).detach()
    v = torch.func.jvp(attention, (Y,), (u,))[0]
    v = func_vjp(v)[0]
    mu = torch.dot(u.flatten(), v.flatten()).detach()
    u = v
    u /= torch.norm(u)

    while torch.abs(mu - mus[-1]) > threshold and i < max_iter:
        i += 1
        mus.append(mu)
        # u and v are always tensors, not tuples
        v = torch.func.jvp(attention, (Y,), (u,))[0]
        v = func_vjp(v)[0]
        mu = torch.dot(u.flatten(), v.flatten()).detach()
        u = v.detach()
        u /= torch.norm(u)
        if i == max_iter:
            print("WARNING: max_iter reached")

    mus.append(mu)
    return mus[-1] ** 0.5


layers = torch.load("bert_decoder/datasets/layers.torch")

#### Alice in Wonderland

ns = torch.load("bert_decoder/datasets/alice/ns.torch")

for b in layers:
    block = torch.load(f"bert_decoder/datasets/block_{b}.torch")

    def attention(input):
        return outs(block, input)

    to_plot = []

    for n in ns:
        radius = []
        means = []
        lipschitz_csts = []
        dataset_in = torch.load(f"bert_decoder/datasets/alice/dataset_{b}_n={n}.torch")

        for X in dataset_in:
            X = X[None, :, :]
            L = loc_lipschitz(attention, X)
            lipschitz_csts.append(L)
            X = X.detach().numpy()
            R = np.sqrt(np.mean(np.linalg.norm(X, axis=2) ** 2))
            mean = np.linalg.norm(np.sum(X, axis=2))
            radius.append(R)
            means.append(mean)
        to_plot.append((radius, lipschitz_csts))

    torch.save(
        torch.Tensor(np.array(to_plot)),
        f"bert_decoder/datasets/alice/to_plot_{b}.torch",
    )


#### AG_NEWS

ns = torch.load("bert_decoder/datasets/AG_NEWS/ns.torch")

for b in layers:
    block = torch.load(f"bert_decoder/datasets/block_{b}.torch")

    def attention(input):
        return outs(block, input)

    to_plot = []

    for n in ns:
        radius = []
        means = []
        lipschitz_csts = []
        dataset_in = torch.load(
            f"bert_decoder/datasets/AG_NEWS/dataset_{b}_n={n}.torch"
        )

        for X in dataset_in:
            X = X[None, :, :]
            L = loc_lipschitz(attention, X)
            lipschitz_csts.append(L)
            X = X.detach().numpy()
            R = np.sqrt(np.mean(np.linalg.norm(X, axis=2) ** 2))
            mean = np.linalg.norm(np.sum(X, axis=2))
            radius.append(R)
            means.append(mean)
        to_plot.append((radius, lipschitz_csts))

    torch.save(
        torch.Tensor(np.array(to_plot)),
        f"bert_decoder/datasets/AG_NEWS/to_plot_{b}.torch",
    )

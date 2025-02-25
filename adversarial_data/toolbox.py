import torch
import numpy as np


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


def retrieve_params(model, name_model, layer):
    if name_model == "bert":
        queries = []
        for name, mod in model.named_modules():
            if name == f"encoder.layer.{layer}.attention.self.query":
                iterator = mod.parameters()
                queries.append(next(iterator).detach())
                queries.append(next(iterator).detach())

        keys = []
        for name, mod in model.named_modules():
            if name == f"encoder.layer.{layer}.attention.self.key":
                iterator = mod.parameters()
                keys.append(next(iterator).detach())
                keys.append(next(iterator).detach())

        values = []
        for name, mod in model.named_modules():
            if name == f"encoder.layer.{layer}.attention.self.value":
                iterator = mod.parameters()
                values.append(next(iterator).detach())
                values.append(next(iterator).detach())

        return queries, keys, values
    elif name_model == "GPT-2":
        d = 768
        param = (model.h[layer].attn.c_attn.weight).detach()
        queries, keys, values = param.split(d, dim=1)
        return queries.detach(), keys.detach(), values.detach()


def compute_A(name_model, h, queries, keys, values):
    if name_model == "bert":
        dim_per_head = 64
        Q = queries[0][h * dim_per_head : (h + 1) * dim_per_head, :].detach()
        K = keys[0][h * dim_per_head : (h + 1) * dim_per_head, :].detach()
        A = torch.matmul(K.T, Q) / np.sqrt(dim_per_head)
        V = values[0][h * dim_per_head : (h + 1) * dim_per_head, :].detach()
        return A, V
    elif name_model == "GPT-2":
        d, dim_per_head = 768, 64
        Q = queries[h * dim_per_head : (h + 1) * dim_per_head, :]  # probably wrong
        K = keys[h * dim_per_head : (h + 1) * dim_per_head, :]
        A = torch.matmul(K.T, Q) / np.sqrt(dim_per_head)
        V = values[h * dim_per_head : (h + 1) * dim_per_head, :]
        return A, V


def compute_bound(n, input, queries, keys, values, model_name):
    num_heads = 12
    r = 0
    for i in range(n):
        for j in range(i + 1, n):
            new = torch.linalg.norm(input[0, i, :] - input[0, j, :])
            if new > r:
                r = new
    R = np.zeros(num_heads)
    for h in range(num_heads):
        A_h, V_h = compute_A(model_name, h, queries, keys, values)
        R_h = 0
        for i in range(n):
            new = torch.linalg.norm(torch.matmul(A_h, input[0, i, :]))
            if new > R_h:
                R_h = new
        R[h] = R_h
    bound = 0
    for h in range(num_heads):
        A_h, V_h = compute_A(model_name, h, queries, keys, values)
        factor_1 = torch.linalg.norm(V_h, ord=2)
        factor_2 = torch.sqrt(R[h] ** 2 * r**2 * (n + 1) + n)
        bound += factor_1 * factor_2
    return bound


def adversarial_data(
    model_name,
    A,
    ns,
    R,
    d=768,
    output_bound=False,
    queries=None,
    keys=None,
    values=None,
):
    data = []
    bound = []
    eigval, eigvec = np.linalg.eig(A)
    eigval, eigvec = eigval[np.isreal(eigval)], eigvec[:, np.isreal(eigval)]
    if eigval.size == 0:
        print("No real eigenvalues!")
    lambda_max, lambda_min = np.max(eigval), np.min(eigval)
    if lambda_max > -8 * lambda_min:
        u = eigvec[:, np.argmax(eigval)]
        for n in ns:
            X = torch.zeros(1, n, d)
            X[0, 0, :] = torch.tensor(u)
            X[0, 1:, :] = torch.tensor(u / 2).repeat(n - 1, 1)
            data.append(R * X)
            if output_bound:
                bound.append(compute_bound(n, R * X, queries, keys, values, model_name))
    else:
        u = eigvec[:, np.argmin(eigval)]
        for n in ns:
            X = torch.zeros(1, n, d)
            X[0, 0, :] = torch.tensor(-u)
            X[0, 1:, :] = torch.tensor(u).repeat(n - 1, 1)
            data.append(R * X)
            if output_bound:
                bound.append(compute_bound(n, R * X, queries, keys, values, model_name))
    if output_bound:
        return data, bound
    else:
        return data


def get_lipschitz_list(model, layer, head, model_name, ns, R):
    model.eval()
    queries, keys, values = retrieve_params(model, model_name, layer)
    A, V = compute_A(model_name, head, queries, keys, values)
    data, bound = adversarial_data(
        model_name, A, ns, R, 768, True, queries, keys, values
    )

    if model_name == "bert":
        attention_block = model.encoder.layer[layer].attention.self
    elif model_name == "GPT-2":

        def attention_block(input):
            return model.h[layer].attn(input)[0]

    lip_list = []
    for X in data:
        L = loc_lipschitz(attention_block, X)  # compute local Lipschitz constants
        X = X.detach().numpy()
        lip_list.append(L)

    return lip_list, bound

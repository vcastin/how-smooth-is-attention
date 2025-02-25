import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from transformers import pipeline
from datasets import load_dataset
from pathlib import Path

Path("bert_encoder/datasets/alice").mkdir(parents=True, exist_ok=True)
Path("bert_encoder/datasets/AG_NEWS").mkdir(parents=True, exist_ok=True)


### Import the model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

### Set forward hooks
activations = []


def print_activations_hook(module, input, output):
    activations.append(input[0])


layers = [0, 6]
torch.save(torch.Tensor(layers).long(), f"bert_encoder/datasets/layers.torch")

for b in layers:
    # choice of the self-attention module
    block = model.encoder.layer[b]
    torch.save(block, f"bert_encoder/datasets/block_{b}.torch")
    block.register_forward_hook(print_activations_hook)


### Build dataset from AG_NEWS

## build raw text
from torchtext.datasets import AG_NEWS

data_iter = AG_NEWS(split="test")
data_list = list(data_iter)  # labeled data
batch_size = 10000

dataloader = torch.utils.data.DataLoader(
    data_list,
    batch_size=batch_size,
)

dataiter = iter(dataloader)

_, text = next(dataiter)
full_text = " ".join(text)  # long string to cut in several pieces

## create dataset
encoded_input = tokenizer(full_text, return_tensors="pt")

ns = range(2, 512, 10)  # choice of sequence lengths
torch.save(torch.Tensor(ns).long(), "bert_encoder/datasets/AG_NEWS/ns.torch")

batch_size = 10  # number of inputs per sequence length

current = 0
for n in ns:
    token_type_ids = torch.zeros(batch_size, n).long()
    attention_mask = torch.ones(batch_size, n).long()
    input_ids = encoded_input["input_ids"][
        :, current : current + (batch_size * n)
    ].reshape(batch_size, n)

    current += batch_size * n

    reshaped_input = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }
    # reshaped_input contains batch_size inputs with of length n

    activations = []
    output = model(**reshaped_input)

    for i, b in enumerate(layers):
        dataset_in = activations[i]
        torch.save(dataset_in, f"bert_encoder/datasets/AG_NEWS/dataset_{b}_n={n}.torch")


#### Build dataset from Alice in Wonderland

## import raw text
import nltk

nltk.download("gutenberg")
from nltk.corpus import gutenberg

text = gutenberg.raw("carroll-alice.txt")

## create dataset
encoded_input = tokenizer(text, return_tensors="pt")

ns = range(2, 100, 2)  # choice of sequence lengths
torch.save(torch.Tensor(ns).long(), f"bert_encoder/datasets/alice/ns.torch")

batch_size = 10  # number of inputs per sequence length

current = 0
for n in ns:
    print(n)
    token_type_ids = torch.zeros(batch_size, n).long()
    attention_mask = torch.ones(batch_size, n).long()
    if (
        encoded_input["input_ids"][:, current : current + (batch_size * n)].shape[0]
        < n * batch_size
    ):
        current = 0
    input_ids = encoded_input["input_ids"][
        :, current : current + (batch_size * n)
    ].reshape(batch_size, n)

    current += batch_size * n

    reshaped_input = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }
    # reshaped_input contains batch_size inputs with of length n

    activations = []
    output = model(**reshaped_input)

    for i, b in enumerate(layers):
        dataset_in = activations[i]
        torch.save(dataset_in, f"bert_encoder/datasets/alice/dataset_{b}_n={n}.torch")

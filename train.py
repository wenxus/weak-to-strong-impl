import itertools
import os
import pickle
import time
from dataclasses import dataclass
import functools
from random import Random
from typing import Callable, Optional

import datasets
from datasets import Dataset as HfDataset
from datasets import load_dataset as hf_load_dataset
import numpy as np
import torch as t
from torch import nn
# import torch_optimizer as toptim
from transformers.modeling_utils import load_sharded_checkpoint

# prep dataset

def tokenize_dataset(
    raw_ds: HfDataset,
    tokenizer: Callable,
    max_ctx: int,
):
    """
    This function prepares the dataset for training. It takes the raw dataset, a formatting function,
    a tokenizer, a maximum context length

    Parameters:
    raw_ds: The raw dataset to be processed.
    tokenizer: The tokenizer to be used on the formatted dataset.
    max_ctx: The maximum context length for the tokenizer.

    Returns:
    ds: The processed and shuffled dataset ready for training.
    """

    def process_function(res):
        toks = tokenizer(res["txt"])
        return dict(
            input_ids=toks["input_ids"],
        )

    ds = raw_ds.map(process_function, batched=False).filter(lambda x: len(x["input_ids"]) < max_ctx)
    return ds

def hf_loader(*hf_name, split_names=None):
    if split_names is None:
        split_names = dict()
    return lambda split: hf_load_dataset(*hf_name, split=split_names.get(split, split))

def format_sciq(ex, rng):
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        ans = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])
    txt = f"Q: {ex['question']} A: {ans}"
    return dict(txt=txt, hard_label=hard_label)

def load_dataset(ds_name: str, seed: int = 0, split_sizes: Optional[dict] = None):
    if split_sizes is None:
        split_sizes = dict(train=None, test=None)

    results = {}
    loader  = hf_loader("sciq")
    for split, n_docs in split_sizes.items():

        ds = loader(split)
        try:
            ds = ds.select(range(n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(functools.partial(format_sciq, rng=Random(seed)))
        ds = ds.map(
            lambda ex: {"soft_label": [1 - float(ex["hard_label"]), float(ex["hard_label"])]}
        )
        ds = ds.shuffle(seed=seed)  # shuffling a bit pointless for test set but wtv
        results[split] = ds
    return results

#%%
ds_name = "sciq"

ds = load_dataset(ds_name, split_sizes=dict(train=500, test=10))
train = list(ds['train'])
test = list(ds['test'])
#%%
print(len(test))
print(f"{test[0].keys()=}")
print(np.mean([x['hard_label'] for x in train]))

#%%
# modify model to do classification
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

inputs = tokenizer("I did not expect", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
logits = outputs.logits
print(logits)
print(tokenizer.batch_decode(outputs.logits.argmax(dim=-1)))

#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
transformer = model.transformer

inputs = tokenizer("I did not expect", return_tensors="pt")
outputs = transformer(**inputs)

print(outputs.last_hidden_state.shape)
#%%
class Linear(nn.Module):
    def __init__(self, d_in, d_out, initializer_range):
        super().__init__()
        self.w = nn.Parameter(t.empty(d_in, d_out))
        self.b = nn.Parameter(t.empty(d_out))
        nn.init.normal_(self.w, std=initializer_range)
        nn.init.normal_(self.b, std=initializer_range)
    
    def forward(self, tokens):
        return tokens * self.w + self.b

class TransformerWithHead(nn.Module):
    def __init__(self, name):
        super().__init__()
        config = AutoConfig.from_pretrained("openai-community/"+name)
        print(f"{config=}")
        # self.tokenizer = AutoModelForCausalLM.from_pretrained("openai-community/"+name)
        model = AutoModelForCausalLM.from_pretrained("openai-community/"+name)
        self.transformer = model.transformer
        self.linear = Linear(getattr(config, "n_embd", getattr(config, "hidden_size", None)), 2, config.initializer_range)

    def forward(self, tokens):
        outputs = transformer(tokens)
        last_hidden_state = outputs.last_hidden_state
        return self.linear(last_hidden_state)

#%% loss
def loss(logits: t.Tensor, targets: t.Tensor) -> t.Tensor:
    loss = t.nn.functional.cross_entropy(logits, targets)
    return loss.mean()


#%%
MODEL_CONFIGS = [
    ModelConfig(
        name="gpt2",
        default_lr=5e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-medium",
        default_lr=5e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-large",
        default_lr=1e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-xl",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
    )]
# train the weak model

def train_model(
    model: t.nn.Module,
    ds: datasets.Dataset,
    batch_size: int,
    lr: float = 1e-5,
    log_every: int = 10,
    eval_every: int = 100,
    eval_batch_size: int = 256,
    minibatch_size: int = 8,
    eval_ds: Optional[datasets.Dataset] = None,
    gradient_checkpointing: bool = False,
    train_with_dropout: bool = False,
    epochs: int = 1,
):
    seed = 0
    random.seed(seed)
    print("Training: LR", lr, "batch_size", batch_size, "minibatch_size", minibatch_size)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    dataset = load_dataset(ds_name, seed=seed, split_sizes=dict(train=20000, test=n_test_docs))

    # Split the training dataset in half
    train_dataset, eval_ds = dataset["train"], dataset["test"] # eval ds is shared for weak and strong models
    # data
    # split og dataset in half
    # train the weak model on the first half
    # get predictions on the second half, save the labels
    # evaluate metrics becomes agreement, also accuracy on test set

    split_data = train_dataset.train_test_split(test_size=0.5, seed=seed)
    train1_ds, train2_ds = split_data["train"], split_data["test"]
    


    nsteps = len(ds) * epochs // batch_size
    model.train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, nsteps)
    step = 0
    # it = itertools.chain.from_iterable(itertools.repeat(ds, epochs))
    losses = []
    accuracies = []
    eval_acc_dict = {}

    train_ds =
    eval_ds = 

    for epoch in epochs:
        model.train()
        


    

# save the label

# train the strong model

# eval

# %%
import anthropic
from anthropic import Anthropic

client = Anthropic(api_key="")
# message = client.messages.create(
#     model="claude-3-5-sonnet-20241022",
#     max_tokens=1024,
#     messages=[
#         {"role": "user", "content": "Hello, world"}
#     ]
# )
# print(message.content)
stream = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Hello, Claude",
        }
    ],
    model="claude-3-opus-20240229",
    stream=True,
)
for event in stream:
    print(event.type)

# %%


# GPT from scratch

This project is a minimal GPT-style language model built entirely from scratch in PyTorch.
It follows the core architecture of the Transformer (multi-head self-attention + feed-forward blocks + residual connections + layer norm), trained on a text dataset to generate new sequences.

I got the code from a tutorial but to make it feel like I didn't just vibe code this, I added my own explanation + understanding in a .ipynb file which you can check out if you want!
## Acknowledgements

 - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
 - [Tutorial Followed](https://youtu.be/kCc8FmEb1nY?si=3mqvtAwFVgfCzk9H)

## Hyperparameters

(You can tweak these at the top of the script)

batch_size = 16 → number of sequences per batch

block_size = 32 → maximum context length

n_embed = 64 → embedding dimension

n_head = 4 → number of attention heads

n_layer = 4 → number of transformer blocks

dropout = 0.0 → dropout rate

learning_rate = 1e-3 → optimizer learning rate

max_iters = 5000 → training iterations
## Run Locally

Clone the project

```bash
  git clone https://github.com/Aruniaaa/GPT-from-scratch
```

Go to the project directory

```bash
  cd GPT-from-scratch
```

Install dependencies

```bash
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Run the code

```bash
  python bigram.py
```

NOTE - The pip install command to download Pytorch on your machine can be different. To check the compatibility of your device and a Pytorch version, [click here.](https://pytorch.org/get-started/locally/)
## Explanation/Documentation

The Google Colab notebook (GPT-from-scratch.ipynb) contains:

A breakdown of the code

A short explanation of how transformers work

I’m still learning, so my explanations might not be perfect, but feedback and corrections are always welcome!
## Project Structure
```
├── Del-data.txt # dataset
├── GPT-from-scratch.ipynb # code + explanation
├── README.md
└── bigram.py  # main training and model script            

```

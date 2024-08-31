# Budget Scaling Monosemanticity

## What is this repo?
- My smaller scale implementation of Anthropic's Scaling Monosemanticity
- Costs less then $10 of compute to train a small LLM and SAE on an RTX 4090, only takes a few hrs
- Creates thousands of interpretable features
- Warning: work in progress, messy code
## Background
A few months ago, I read [Antrhopic's Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) blog post, where they described how they extracted tens of millions of abstract, interpretable features from Claude 3 Sonnet, using a sparse autoencoder (SAE). This allowed them to better explain how the model worked and better control it.

Immediately, I wanted to implement this myself. However, with my compute budget of approximately $0, I was a few million(?) dollars short of the required budget to reproduce the paper on a full-scale LLM. 

Enter [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

This dataset of a few million synthetically generated children's stories is very easy for LLMs to train on, and allows a model with only a few **million** parameters to produce coherent English and basic reasoning. I trained a small decoder-only transformer (GPT) on TinyStories, and after a few hours of training on an RTX 4090, it was able to write some short stories that made sense for the most part. Because the model was so small, I was able to cheaply extract about 50GB of residuals, which I then used to train an SAE with about 16k possible features.

## Results
Due to the relatively low variability of TinyStories and small LLM, the SAE was able to model the residual stream fairly well (R2 of 77%, about 20 active features for a given embedding) compared to Anthropic's full-scale implementation (R^2 of 65%, 300 active features). More importantly, the features were actually interpretable! For example, here are some of the top activating tokens over a small subset of the data, for feature 93:

> there was a fish named Nemo. Nemo loved to swim in the

...

> to the beach and asked her mom if she could swim in the ocean. Her mom said to be

...

> One day, the octopus and the fish found a big jug. They wanted to pour


As you can see, this feature corresponds to things related the ocean/water! I haven't yet tried steering story generation using features, but that's next.

## How to use
This repo is pretty hacky right now. It consists of a few jupyter notebooks that you need to run in the right order. In the future I'll clean it up + improve code quality, but for now, I'm still busy running experiments, so actually solidifying the notebooks into python scripts would be detrimental. With that said, here are some instructions for running the code:

1. Rent (or own) a computer with an RTX 4090. The notebook is designed to use about 24 GB of vram. This could be reduced easily by never loading the SAE and LLM at the same time, but for now, about 24 GB is the minimum vram requirement.
2. Download tinystories-train.txt into this directory from [tinystories on huggingface](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main)
3. Use pip to install the dependencies. There isn't anything special here, just torch, numpy, matplotlib, huggingface tokenizers, wandb (weights and biases).
4. Train the tokenizer + tokenize the dataset by running `train-tokenizer.ipynb` (takes about 10 minutes)
5. Train the LLM by running `train-llm.ipynb` (training takes a few hrs, you can stop after about an hour and still get reasonable results)
6. Generate 50 GB of residuals from the model by running `generate-residuals.ipynb` (10 mins)
7. Train SAE using `train-sae.ipynb` (takes a few hrs, but you can stop after 1-2 hrs and get reasonable results)
8. Check the number of active features and other stats using `eval-sae.ipynb`
9. Interpret features in `sae-interpretability.ipynb`

"""Utility functions."""

import torch
import torch.nn.functional as F
import numpy as np

EPS = 1e-17
NEG_INF = -1e30


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def gumbel_sample(shape):
    """Sample Gumbel noise."""
    uniform = torch.rand(shape).float()
    return - torch.log(EPS - torch.log(uniform + EPS))


def gumbel_softmax_sample(logits, temp=1.):
    """Sample from the Gumbel softmax / concrete distribution."""
    gumbel_noise = gumbel_sample(logits.size())
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    return F.softmax((logits + gumbel_noise) / temp, dim=-1)


def gaussian_sample(mu, log_var):
    """Sample from Gaussian distribution."""
    gaussian_noise = torch.randn(mu.size())
    return mu + torch.exp(log_var * 0.5) * gaussian_noise


def kl_gaussian(mu, log_var):
    """KL divergence between Gaussian posterior and standard normal prior."""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)


def kl_categorical_uniform(preds):
    """KL divergence between categorical distribution and uniform prior."""
    kl_div = preds * torch.log(preds + EPS)  # Up to constant, can be negative.
    return kl_div.sum(1)


def kl_categorical(preds, log_prior):
    """KL divergence between two categorical distributions."""
    kl_div = preds * (torch.log(preds + EPS) - log_prior)
    return kl_div.sum(1)


def poisson_categorical_log_prior(length, rate):
    """Categorical prior populated with log probabilities of Poisson dist."""
    values = torch.arange(1, length + 1, dtype=torch.float32).unsqueeze(0)
    return np.log(rate) * values - rate - (values + 1).lgamma()


def log_cumsum(probs, dim=1):
    """Calculate log of inclusive cumsum."""
    return torch.log(torch.cumsum(probs, dim=dim) + EPS)


def generate_toy_data(num_symbols=4, num_segments=3, max_segment_len=5):
    """Generate toy data sample with repetition of symbols (EOS symbol: 0)."""
    seq = []
    symbols = np.random.choice(np.arange(1, num_symbols), 3, replace=False)
    for seg_id in range(num_segments):
        segment_len = np.random.choice(np.arange(1, max_segment_len))
        seq += [symbols[seg_id]] * segment_len
    seq += [0]
    return torch.tensor(seq, dtype=torch.int64)[None, :]

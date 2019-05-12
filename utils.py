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
    if mu.is_cuda:
        gaussian_noise = gaussian_noise.cuda()
    return mu + torch.exp(log_var * 0.5) * gaussian_noise


def kl_gaussian(mu, log_var):
    """KL divergence between Gaussian posterior and standard normal prior."""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)


def kl_categorical_uniform(preds):
    """KL divergence between categorical distribution and uniform prior."""
    kl_div = preds * torch.log(preds + EPS)  # Constant term omitted.
    return kl_div.sum(1)


def kl_categorical(preds, log_prior):
    """KL divergence between two categorical distributions."""
    kl_div = preds * (torch.log(preds + EPS) - log_prior)
    return kl_div.sum(1)


def poisson_categorical_log_prior(length, rate, device):
    """Categorical prior populated with log probabilities of Poisson dist."""
    rate = torch.tensor(rate, dtype=torch.float32, device=device)
    values = torch.arange(
        1, length + 1, dtype=torch.float32, device=device).unsqueeze(0)
    log_prob_unnormalized = torch.log(
        rate) * values - rate - (values + 1).lgamma()
    # TODO(tkipf): Length-sensitive normalization.
    return F.log_softmax(log_prob_unnormalized, dim=1)  # Normalize.


def log_cumsum(probs, dim=1):
    """Calculate log of inclusive cumsum."""
    return torch.log(torch.cumsum(probs, dim=dim) + EPS)


def generate_toy_data(num_symbols=5, num_segments=3, max_segment_len=5):
    """Generate toy data sample with repetition of symbols (EOS symbol: 0)."""
    seq = []
    symbols = np.random.choice(
        np.arange(1, num_symbols + 1), num_segments, replace=False)
    for seg_id in range(num_segments):
        segment_len = np.random.choice(np.arange(1, max_segment_len))
        seq += [symbols[seg_id]] * segment_len
    seq += [0]
    return torch.tensor(seq, dtype=torch.int64)


def get_lstm_initial_state(batch_size, hidden_dim, device):
    """Get empty (zero) initial states for LSTM."""
    hidden_state = torch.zeros(batch_size, hidden_dim, device=device)
    cell_state = torch.zeros(batch_size, hidden_dim, device=device)
    return hidden_state, cell_state


def get_segment_probs(all_b_samples, all_masks, segment_id):
    """Get segment probabilities for a particular segment ID."""
    neg_cumsum = 1 - torch.cumsum(all_b_samples[segment_id], dim=1)
    if segment_id > 0:
        return neg_cumsum * all_masks[segment_id - 1]
    else:
        return neg_cumsum


def get_losses(inputs, outputs, args, beta_b=.1, beta_z=.1, prior_rate=3.,):
    """Get losses (NLL, KL divergences and neg. ELBO).

    Args:
        inputs: Padded input sequences.
        outputs: CompILE model output tuple.
        args: Argument dict from `ArgumentParser`.
        beta_b: Scaling factor for KL term of boundary variables (b).
        beta_z: Scaling factor for KL term of latents (z).
        prior_rate: Rate (lambda) for Poisson prior.
    """

    targets = inputs.view(-1)
    all_encs, all_recs, all_masks, all_b, all_z = outputs
    input_dim = args.num_symbols + 1

    nll = 0.
    kl_z = 0.
    for seg_id in range(args.num_segments):
        seg_prob = get_segment_probs(
            all_b['samples'], all_masks, seg_id)
        preds = all_recs[seg_id].view(-1, input_dim)
        seg_loss = F.cross_entropy(
            preds, targets, reduction='none').view(-1, inputs.size(1))

        # Ignore EOS token (last sequence element) in loss.
        nll += (seg_loss[:, :-1] * seg_prob[:, :-1]).sum(1).mean(0)

        # KL divergence on z.
        if args.latent_dist == 'gaussian':
            mu, log_var = torch.split(
                all_z['logits'][seg_id], args.latent_dim, dim=1)
            kl_z += kl_gaussian(mu, log_var).mean(0)
        elif args.latent_dist == 'concrete':
            kl_z += kl_categorical_uniform(
                F.softmax(all_z['logits'][seg_id], dim=-1)).mean(0)
        else:
            raise ValueError('Invalid argument for `latent_dist`.')

    # KL divergence on b (first segment only, ignore first time step).
    # TODO(tkipf): Implement alternative prior on soft segment length.
    probs_b = F.softmax(all_b['logits'][0], dim=-1)
    log_prior_b = poisson_categorical_log_prior(
        probs_b.size(1), prior_rate, device=inputs.device)
    kl_b = args.num_segments * kl_categorical(
        probs_b[:, 1:], log_prior_b[:, 1:]).mean(0)

    loss = nll + beta_z * kl_z + beta_b * kl_b
    return loss, nll, kl_z, kl_b


def get_reconstruction_accuracy(inputs, outputs, args):
    """Calculate reconstruction accuracy (averaged over sequence length)."""

    all_encs, all_recs, all_masks, all_b, all_z = outputs

    batch_size = inputs.size(0)

    rec_seq = []
    rec_acc = 0.
    for sample_idx in range(batch_size):
        prev_boundary_pos = 0
        rec_seq_parts = []
        for seg_id in range(args.num_segments):
            boundary_pos = torch.argmax(
                all_b['samples'][seg_id], dim=-1)[sample_idx]
            if prev_boundary_pos > boundary_pos:
                boundary_pos = prev_boundary_pos
            seg_rec_seq = torch.argmax(all_recs[seg_id], dim=-1)
            rec_seq_parts.append(
                seg_rec_seq[sample_idx, prev_boundary_pos:boundary_pos])
            prev_boundary_pos = boundary_pos
        rec_seq.append(torch.cat(rec_seq_parts))
        cur_length = rec_seq[sample_idx].size(0)
        matches = rec_seq[sample_idx] == inputs[sample_idx, :cur_length]
        rec_acc += matches.float().mean()
    rec_acc /= batch_size
    return rec_acc, rec_seq

import torch
import torch.nn.functional as F
from torch import nn

import utils


class CompILE(nn.Module):
    """CompILE example implementation.

    Args:
        input_dim: Dictionary size of embeddings.
        hidden_dim: Number of hidden units.
        latent_dim: Dimensionality of latent variables (z).
        max_num_segments: Maximum number of segments to predict.
        temp_b: Gumbel softmax temperature for boundary variables (b).
        temp_z: Temperature for latents (z), only if latent_dist='concrete'.
        beta_b: Scaling factor for KL term of boundary variables (b).
        beta_z: Scaling factor for KL term of latents (z).
        prior_rate: Rate (lambda) for Poisson prior.
        latent_dist: Whether to use Gaussian latents ('gaussian') or concrete /
            Gumbel softmax latents ('concrete').
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_segments,
                 temp_b=1., temp_z=1., beta_b=.1, beta_z=.1, prior_rate=3.,
                 latent_dist='gaussian'):
        super(CompILE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_num_segments = max_num_segments
        self.temp_b = temp_b
        self.temp_z = temp_z
        self.beta_b = beta_b
        self.beta_z = beta_z
        self.prior_rate = prior_rate
        self.latent_dist = latent_dist

        self.embed = nn.Embedding(input_dim, hidden_dim)
        self.lstm_cell = nn.LSTMCell(hidden_dim, hidden_dim)

        # LSTM output heads.
        self.head_z_1 = nn.Linear(hidden_dim, hidden_dim)  # Latents (z).

        if latent_dist == 'gaussian':
            self.head_z_2 = nn.Linear(hidden_dim, latent_dim * 2)
        elif latent_dist == 'concrete':
            self.head_z_2 = nn.Linear(hidden_dim, latent_dim)

        self.head_b_1 = nn.Linear(hidden_dim, hidden_dim)  # Boundaries (b).
        self.head_b_2 = nn.Linear(hidden_dim, 1)

        # Decoder MLP.
        self.decode_1 = nn.Linear(latent_dim, hidden_dim)
        self.decode_2 = nn.Linear(hidden_dim, input_dim)

    def get_lstm_initial_state(self, batch_size, cuda=False):
        """Get empty (zero) initial states for LSTM."""
        hidden_state = torch.zeros(batch_size, self.hidden_dim)
        cell_state = torch.zeros(batch_size, self.hidden_dim)
        if cuda:
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()
        return hidden_state, cell_state

    def masked_encode(self, inputs, mask):
        """Run masked RNN encoder on input sequence."""
        hidden = self.get_lstm_initial_state(inputs.size(0), inputs.is_cuda)
        outputs = []
        for step in range(inputs.size(1)):
            hidden = self.lstm_cell(inputs[:, step], hidden)
            hidden = (mask[:, step, None] * hidden[0],
                      mask[:, step, None] * hidden[1])  # Apply mask.
            outputs.append(hidden[0])
        return torch.stack(outputs, dim=1)

    def get_boundaries(self, encodings, segment_id, lengths, evaluate=False):
        """Get boundaries (b) for a single segment in batch."""
        if segment_id == self.max_num_segments - 1:
            # Last boundary is always placed on last sequence element.
            logits_b = None
            sample_b = torch.zeros_like(encodings[:, :, 0]).scatter_(
                1, lengths.unsqueeze(1) - 1, 1)
        else:
            hidden = F.relu(self.head_b_1(encodings))
            logits_b = self.head_b_2(hidden).squeeze(-1)
            # Mask out first position with large neg. value.
            neg_inf = torch.ones(
                encodings.size(0), 1, device=encodings.device) * utils.NEG_INF
            # TODO(tkipf): Mask out padded positions with large neg. value.
            logits_b = torch.cat([neg_inf, logits_b[:, 1:]], dim=1)
            if not evaluate:
                sample_b = utils.gumbel_softmax_sample(
                    logits_b, temp=self.temp_b)
            else:
                sample_b_idx = torch.argmax(logits_b, dim=1)
                sample_b = utils.to_one_hot(sample_b_idx, logits_b.size(1))

        return logits_b, sample_b

    def get_latents(self, encodings, probs_b, evaluate=False):
        """Read out latents (z) form input encodings for a single segment."""
        readout_mask = probs_b[:, 1:, None]  # Offset readout by 1 to left.
        readout = (encodings[:, :-1] * readout_mask).sum(1)
        hidden = F.relu(self.head_z_1(readout))
        logits_z = self.head_z_2(hidden)

        # Gaussian latents.
        if self.latent_dist == 'gaussian':
            if not evaluate:
                mu, log_var = torch.split(logits_z, self.latent_dim, dim=1)
                sample_z = utils.gaussian_sample(mu, log_var)
            else:
                sample_z = logits_z[:, :self.latent_dim]

        # Concrete / Gumbel softmax latents.
        elif self.latent_dist == 'concrete':
            if not evaluate:
                sample_z = utils.gumbel_softmax_sample(
                    logits_z, temp=self.temp_z)
            else:
                sample_z_idx = torch.argmax(logits_z, dim=1)
                sample_z = utils.to_one_hot(sample_z_idx, logits_z.size(1))

        return logits_z, sample_z

    def decode(self, sample_z, length):
        """Decode single time step from latents and repeat over full seq."""
        hidden = F.relu(self.decode_1(sample_z))
        pred = self.decode_2(hidden)
        return pred.unsqueeze(1).repeat(1, length, 1)

    def get_next_masks(self, all_b_samples):
        """Get RNN hidden state masks for next segment."""
        if len(all_b_samples) < self.max_num_segments:
            # Product over cumsums (via log->sum->exp).
            log_cumsums = list(
                map(lambda x: utils.log_cumsum(x, dim=1), all_b_samples))
            mask = torch.exp(sum(log_cumsums))
            return mask
        else:
            return None

    def get_segment_probs(self, all_b_samples, all_masks, segment_id):
        """Get segment probabilities for a particular segment ID."""
        neg_cumsum = 1 - torch.cumsum(all_b_samples[segment_id], dim=1)
        if segment_id > 0:
            return neg_cumsum * all_masks[segment_id - 1]
        else:
            return neg_cumsum

    def get_losses(self, inputs, lengths):
        """Get losses (NLL, KL divergences and neg. ELBO)."""
        targets = inputs.view(-1)
        all_encs, all_recs, all_masks, all_b, all_z = self.forward(
            inputs, lengths)

        nll = 0.
        kl_z = 0.
        for seg_id in range(self.max_num_segments):
            seg_prob = self.get_segment_probs(
                all_b['samples'], all_masks, seg_id)
            preds = all_recs[seg_id].view(-1, self.input_dim)
            seg_loss = F.cross_entropy(
                preds, targets, reduction='none').view(-1, inputs.size(1))

            # Ignore EOS token (last sequence element) in loss.
            nll += (seg_loss[:, :-1] * seg_prob[:, :-1]).sum(1).mean(0)

            # KL divergence on z.
            if self.latent_dist == 'gaussian':
                mu, log_var = torch.split(
                    all_z['logits'][seg_id], self.latent_dim, dim=1)
                kl_z += utils.kl_gaussian(mu, log_var).mean(0)
            elif self.latent_dist == 'concrete':
                kl_z += utils.kl_categorical_uniform(
                    F.softmax(all_z['logits'][seg_id], dim=-1)).mean(0)

        # KL divergence on b (first segment only, ignore first time step).
        # TODO(tkipf): Implement alternative prior on soft segment length.
        probs_b = F.softmax(all_b['logits'][0], dim=-1)
        log_prior_b = utils.poisson_categorical_log_prior(
            probs_b.size(1), self.prior_rate, device=inputs.device)
        kl_b = self.max_num_segments * utils.kl_categorical(
            probs_b[:, 1:], log_prior_b[:, 1:]).mean(0)

        loss = nll + self.beta_z * kl_z + self.beta_b * kl_b
        return loss, nll, kl_z, kl_b

    def get_reconstruction_accuracy(self, inputs, lengths):
        """Calculate reconstruction accuracy (averaged over sequence length)."""
        all_encs, all_recs, all_masks, all_b, all_z = self.forward(
            inputs, lengths, evaluate=True)

        batch_size = inputs.size(0)

        rec_seq = None
        rec_acc = 0.
        for sample_idx in range(batch_size):
            prev_boundary_pos = 0
            rec_seq_parts = []
            for seg_id in range(self.max_num_segments):
                boundary_pos = torch.argmax(
                    all_b['samples'][seg_id], dim=-1)[sample_idx]
                if prev_boundary_pos > boundary_pos:
                    boundary_pos = prev_boundary_pos
                seg_rec_seq = torch.argmax(all_recs[seg_id], dim=-1)
                rec_seq_parts.append(
                    seg_rec_seq[sample_idx, prev_boundary_pos:boundary_pos])
                prev_boundary_pos = boundary_pos
            rec_seq = torch.cat(rec_seq_parts)
            cur_length = rec_seq.size(0)
            matches = (rec_seq == inputs[sample_idx, :cur_length]).float()
            rec_acc += matches.mean()
        rec_acc /= batch_size
        return rec_acc, rec_seq

    def forward(self, inputs, lengths, evaluate=False):

        # Embed inputs.
        embeddings = self.embed(inputs)

        # Create initial mask.
        mask = torch.ones(
            inputs.size(0), inputs.size(1), device=inputs.device)

        all_b = {'logits': [], 'samples': []}
        all_z = {'logits': [], 'samples': []}
        all_encs = []
        all_recs = []
        all_masks = []
        for seg_id in range(self.max_num_segments):

            # Get masked LSTM encodings of inputs.
            encodings = self.masked_encode(embeddings, mask)
            all_encs.append(encodings)

            # Get boundaries (b) for current segment.
            logits_b, sample_b = self.get_boundaries(
                encodings, seg_id, lengths, evaluate)
            all_b['logits'].append(logits_b)
            all_b['samples'].append(sample_b)

            # Get latents (z) for current segment.
            logits_z, sample_z = self.get_latents(
                encodings, sample_b, evaluate)
            all_z['logits'].append(logits_z)
            all_z['samples'].append(sample_z)

            # Get masks for next segment.
            mask = self.get_next_masks(all_b['samples'])
            all_masks.append(mask)

            # Decode current segment from latents (z).
            reconstructions = self.decode(sample_z, length=inputs.size(1))
            all_recs.append(reconstructions)

        return all_encs, all_recs, all_masks, all_b, all_z

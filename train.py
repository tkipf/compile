import argparse
import torch

import utils
import modules

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=100,
                    help='Number of training iterations.')
parser.add_argument('--learning-rate', type=float, default=1e-2,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--latent-dim', type=int, default=16,
                    help='Dimensionality of latent variables.')
parser.add_argument('--latent-dist', type=str, default='gaussian',
                    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Mini-batch size (for averaging gradients).')

parser.add_argument('--num-symbols', type=int, default=4,
                    help='Number of distinct symbols in data generation.')
parser.add_argument('--num-segments', type=int, default=3,
                    help='Number of segments in data generation.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--log-interval', type=int, default=1,
                    help='Logging interval.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda' if args.cuda else 'cpu')

model = modules.CompILE(
    input_dim=args.num_symbols + 1,  # +1 for EOS symbol.
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim,
    max_num_segments=args.num_segments,
    latent_dist=args.latent_dist).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Train model.
print('Training model...')
for step in range(args.iterations):
    batch_loss = 0
    batch_acc = 0
    optimizer.zero_grad()
    for _ in range(args.batch_size):

        # Generate data.
        data = utils.generate_toy_data(
            num_symbols=args.num_symbols,
            num_segments=args.num_segments)
        data = data.to(device)

        # Run forward pass.
        model.train()
        loss, nll, kl_z, kl_b = model.get_losses(data)

        # Run eval.
        model.eval()
        acc = model.get_reconstruction_accuracy(data)

        # Accumulate metrics.
        batch_acc += acc.item() / args.batch_size
        batch_loss += nll.item() / args.batch_size

        # Accumulate gradients.
        loss = loss / args.batch_size
        loss.backward()

    optimizer.step()
    if step % args.log_interval == 0:
        print('step: {}, nll_train: {:.6f}, rec_acc_eval: {:.3f}'.format(
            step, batch_loss, batch_acc))

# CompILE implementation example

This is an example implementation of the CompILE model on a simple sequence segmentation toy task in PyTorch with minimal dependencies. This implementation is optimized for readability rather than speed/efficiency, and operates on single samples instead of mini-batches of data. 

## Dependencies:
* Python 3.6 or later
* Numpy 1.15 or later
* PyTorch 4.1 or later

## Files:
* `train.py`: Entry point, run `python train.py` to train the model.
* `modules.py`: This file contains the CompILE model in a single PyTorch module.
* `utils.py`: This file contains utilities for the CompILE model and the toy data generator `generate_toy_data()`.

## Task:
We randomly generate sequences of the form `[NUM_1]*FREQ_1 + [NUM_2]*FREQ_2 + [NUM_3]*FREQ_3`, where `NUM_X` and `FREQ_X` are randomly drawn integers from a pre-defined range (with and without replacement, respectively). An example sequence looks as follows: `[4, 4, 5, 5, 5, 3, 3, 3]`. The CompILE model has to identify the correct segmentation (in this case: 3 segments) and encode each segment into a single latent variable, from which the respective segment will be reconstructed. The decoder is a simple two-layer MLP conditioned on the latent variable of the segment which outputs a single integer (as a categorical variable), which we repeat over the full sequence length to compute a loss.

## Running the model:
Simply run `python train.py` to train the model with default settings on CPU. Please inspect `train.py` and `modules.py`

During training, the script prints negative log likelihood (`nll_train`) and evaluation reconstruction accuracy (`rec_acc_eval`) after every training iteration (a single gradient step). `rec_acc_eval` corresponds to the average (per time step) reconstruction accuracy for a mini-batch of generated samples, where the model runs in evaluation mode (concrete latent variables replaced with discrete ones, and Gaussian latents are replaced by their predicted mean). `rec_acc_eval` of `1.00` corresponds to perfect segmentation and reconstruction in this particular task.

The learning rate for Adam is chosen relatively high (`1e-2`) to reduce training time, which can however destabilize training in rare cases (for some random seeds). Please try reducing the learning rate to `1e-3` if you observe this effect. The default setting uses Gaussian latent variables (z) to encode segments. To train the model with concrete / Gumbel softmax latent variables, run `python train.py --latent-dist concrete`.

Example run (`python train.py`):

```
Training model...
step: 0, nll_train: 13.659223, rec_acc_eval: 0.000
input sample: tensor([4, 4, 4, 3, 3, 3, 1, 1, 1, 1])
reconstruction: tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
step: 1, nll_train: 13.301423, rec_acc_eval: 0.264
input sample: tensor([5, 5, 5, 5, 4, 4, 2, 2, 2])
reconstruction: tensor([4, 4, 4, 4, 4, 4, 4, 4, 4])
step: 2, nll_train: 12.695760, rec_acc_eval: 0.373
input sample: tensor([3, 5, 5, 2, 2, 2, 2])
reconstruction: tensor([4, 4, 2, 2, 2, 2, 2])
step: 3, nll_train: 12.221995, rec_acc_eval: 0.462
input sample: tensor([3, 3, 2, 4, 4, 4])
reconstruction: tensor([3, 3, 4, 4, 4, 4])
step: 4, nll_train: 11.990509, rec_acc_eval: 0.600
input sample: tensor([5, 5, 5, 3, 3, 1])
reconstruction: tensor([3, 3, 3, 3, 3, 1])
step: 5, nll_train: 11.107215, rec_acc_eval: 0.611
input sample: tensor([2, 2, 4, 5, 5, 5])
reconstruction: tensor([2, 5, 5, 5, 5, 5])
step: 6, nll_train: 10.142213, rec_acc_eval: 0.674
input sample: tensor([5, 5, 5, 3, 3, 3, 3, 4, 4, 4, 4])
reconstruction: tensor([5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4])
step: 7, nll_train: 9.297485, rec_acc_eval: 0.639
input sample: tensor([5, 5, 5, 5, 3, 4, 4, 4])
reconstruction: tensor([5, 5, 5, 4, 4, 4, 4, 4])
step: 8, nll_train: 8.644855, rec_acc_eval: 0.638
input sample: tensor([5, 5, 5, 5, 2, 2, 3])
reconstruction: tensor([5, 5, 2, 2, 2, 2, 2])
step: 9, nll_train: 7.950069, rec_acc_eval: 0.741
input sample: tensor([2, 5, 5, 5, 5, 3, 3])
reconstruction: tensor([2, 5, 5, 5, 5, 5, 5])
step: 10, nll_train: 7.444612, rec_acc_eval: 0.833
input sample: tensor([1, 1, 1, 4, 4, 5])
reconstruction: tensor([1, 1, 1, 4, 5, 5])
step: 11, nll_train: 6.921782, rec_acc_eval: 0.903
input sample: tensor([1, 2, 2, 2, 2, 5, 5])
reconstruction: tensor([1, 2, 2, 2, 2, 5, 5])
step: 12, nll_train: 6.760701, rec_acc_eval: 0.912
input sample: tensor([4, 2, 5, 5, 5])
reconstruction: tensor([4, 2, 5, 5, 5])
step: 13, nll_train: 5.633484, rec_acc_eval: 0.937
input sample: tensor([2, 4, 4, 4, 4, 5, 5, 5, 5])
reconstruction: tensor([2, 4, 4, 4, 4, 5, 5, 5, 5])
step: 14, nll_train: 6.476520, rec_acc_eval: 0.923
input sample: tensor([5, 5, 1, 1, 4, 4])
reconstruction: tensor([5, 5, 1, 1, 4, 4])
step: 15, nll_train: 4.534423, rec_acc_eval: 0.965
input sample: tensor([2, 2, 2, 4, 4, 4, 4, 1, 1])
reconstruction: tensor([2, 2, 2, 4, 4, 4, 4, 1, 1])
step: 16, nll_train: 4.091990, rec_acc_eval: 0.953
input sample: tensor([4, 4, 5, 5, 5, 5, 1, 1, 1, 1])
reconstruction: tensor([4, 4, 5, 5, 5, 5, 1, 1, 1, 1])
step: 17, nll_train: 3.090653, rec_acc_eval: 0.989
input sample: tensor([5, 5, 4, 4, 4, 2, 2])
reconstruction: tensor([5, 5, 4, 4, 4, 2, 2])
step: 18, nll_train: 2.511392, rec_acc_eval: 0.995
input sample: tensor([2, 2, 2, 4, 3, 3])
reconstruction: tensor([2, 2, 2, 4, 3, 3])
step: 19, nll_train: 1.535790, rec_acc_eval: 0.997
input sample: tensor([4, 4, 4, 3, 3, 1])
reconstruction: tensor([4, 4, 4, 3, 3, 1])
step: 20, nll_train: 1.485223, rec_acc_eval: 0.992
input sample: tensor([2, 2, 2, 3, 3, 1, 1])
reconstruction: tensor([2, 2, 2, 3, 3, 1, 1])
step: 21, nll_train: 1.094403, rec_acc_eval: 0.996
input sample: tensor([3, 3, 3, 3, 1, 1, 1, 4, 4, 4])
reconstruction: tensor([3, 3, 3, 3, 1, 1, 1, 4, 4, 4])
step: 22, nll_train: 0.720283, rec_acc_eval: 0.998
input sample: tensor([1, 1, 1, 1, 2, 2, 4, 4, 4])
reconstruction: tensor([1, 1, 1, 1, 2, 2, 4, 4, 4])
step: 23, nll_train: 0.785404, rec_acc_eval: 0.994
input sample: tensor([1, 1, 1, 5, 5, 4])
reconstruction: tensor([1, 1, 1, 5, 5, 4])
step: 24, nll_train: 0.665226, rec_acc_eval: 1.000
input sample: tensor([5, 3, 4, 4])
reconstruction: tensor([5, 3, 4, 4])
step: 25, nll_train: 0.618572, rec_acc_eval: 1.000
input sample: tensor([5, 5, 5, 5, 2, 2, 1, 1, 1, 1])
reconstruction: tensor([5, 5, 5, 5, 2, 2, 1, 1, 1, 1])
```
# Converge to Surprise

Official code for **"Converge to Surprise: Evolutionary Self-supervised Learning on Images"**
(Canlin Zhang, Xiuwen Liu). Paper: https://arxiv.org/abs/2607.06887

Fully **unsupervised, non-parametric image clustering** (MNIST, FashionMNIST, USPS;
CIFAR-10 supported) — the strictest deep-clustering setting, where the number of
classes *K* is **not** given to the optimizer. A deep network is trained **without
labels** by maximizing a statistical **Surprise Score** with a **gradient-free
Evolution Strategy (ES)**, plus periodic **gradient-based fine-tuning on surrogate
labels** to consolidate clusters. Everything is learned from **raw pixels** — no
pretrained features.

## Idea in one paragraph

Under the Principle of Maximum Entropy, the most conservative null hypothesis
`H0` is that images are i.i.d. pixel noise. Two complementary **chessboard
half-views** of an image share zero mutual information under `H0`. The **Surprise
Score** `S(θ)` measures how strongly the network's two-view argmax assignments
*violate* that independence — i.e. how much non-random structure it found. Because
`S(θ)` reads `argmax` indices over a dynamically-changing set of over-matching
clusters, it is non-differentiable and cannot be reduced to a per-step loss; we
therefore maximize it with an **ES outer loop** (the long-term *explorer*) and a
periodic **surrogate fine-tuning inner loop** (the short-term *consolidator*).

## Repository structure

```
main.py                              # training entry point (ES per epoch + staged fine-tuning)
test.py                              # evaluation: DeepDPM square-Hungarian ACC + NMI/ARI + purity report
scripts/
  deep_network.py                    # ResNet-9 backbone (B,C,H,W) -> (B,K) logits
  pair_maker.py                      # chessboard two-view masking + per-dataset augmentation
  surprise_score.py                  # the training signal (per-dim binary-KL surprise)
  evolution_strategy.py              # parallel ES (torch.vmap), mirrored sampling, centered-rank update
  train_and_eval_one_epoch.py        # 3-phase surrogate fine-tuning
  utils.py                           # DeepDPM square-Hungarian cluster_acc helper
```
(`*_v1.py` / `*_v2.py` under `scripts/` are stale snapshots kept for reference and
are **not** imported.)

## Setup

```bash
pip install -r requirements.txt
```
Requires Python 3.9+ and a CUDA-capable GPU (training is GPU-only; CPU evaluation
works with `CUDA_VISIBLE_DEVICES=""`). The MNIST / FashionMNIST / USPS / CIFAR-10
datasets are **downloaded automatically** by `torchvision` into `./data/` on first
run (this folder is git-ignored).

## Training

Each command runs `--number_of_experiments` (default **5**) independent experiments
into `<save_dir>0/ … 4/`. Pick a GPU with enough free memory (~30 GB at the default
`base_width=8`, `population_size=32`).

```bash
# MNIST — 2-stage schedule (pure ES until 2000, then fine-tuning to 3000)
CUDA_VISIBLE_DEVICES=0 python main.py \
    --save_dir ./model_bin/mnist_run/ --dataset_name MNIST \
    --train_start_epoch_1 2000 --train_start_epoch_2 2000 --num_epochs 3001

# FashionMNIST — same 2-stage schedule
CUDA_VISIBLE_DEVICES=0 python main.py \
    --save_dir ./model_bin/fmnist_run/ --dataset_name FashionMNIST \
    --train_start_epoch_1 2000 --train_start_epoch_2 2000 --num_epochs 3001

# USPS — 3-stage schedule, larger batch, longer training (fewer images)
CUDA_VISIBLE_DEVICES=0 python main.py \
    --save_dir ./model_bin/usps_run/ --dataset_name USPS \
    --train_start_epoch_1 4000 --train_start_epoch_2 8000 --num_epochs 9001 --N 3650
```

Checkpoints are written per experiment: `optimal_autoencoder.pth` (rewritten every
epoch) and frozen `optimal_autoencoder_epoch_<n>.pth` every 1000 epochs.

## Evaluation

Evaluate a single finished experiment directory:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --save_dir ./model_bin/mnist_run/0/ --dataset_name MNIST
```

This reports clustering accuracy (DeepDPM square-Hungarian assignment), NMI, ARI,
the number of active clusters, per-class accuracy, and a cluster-purity report, and
writes `test_results.txt` into the experiment directory.

## Key hyperparameters (defaults)

`--K 64 --N 3000 --base_width 8 --population_size 32 --sigma 0.02
--learning_rate 0.005 --weight_decay 0.005 --num_train_epochs 4 --kl_threshold 0.005
--number_of_experiments 5`. Per-dataset augmentation is configured in
`scripts/pair_maker.py`.

## Citation

```bibtex
@article{zhang2026converge,
  title   = {Converge to Surprise: Evolutionary Self-supervised Learning on Images},
  author  = {Zhang, Canlin and Liu, Xiuwen},
  journal = {arXiv preprint arXiv:2607.06887},
  year    = {2026}
}
```

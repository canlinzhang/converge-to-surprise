"""
End-to-end differentiable baseline: maximize the SOFT surprise score by gradient descent.

This is the "exploitative" counterpart of `main.py`, used for the appendix experiment on
Hypothesis 1. Everything about the data is identical to the SOTA run -- the same
`PairMaker`, hence the same chessboard masking and the same independent per-view
augmentation -- and the network is the same ResNet9. The ONLY thing that changes is the
optimizer and the objective:

    main.py          hard S(theta) (argmax counts)  ->  gradient-free ES  +  surrogate FT
    main_softmax.py  soft S_soft(theta; tau, eps)   ->  Adam, end to end

so `scripts/surprise_score.py`, `scripts/evolution_strategy.py` and
`scripts/train_and_eval_one_epoch.py` play no part in training here. (SurpriseScore is
imported for LOGGING only -- see `hard_diagnostics` -- so that the per-epoch numbers in
train.log mean exactly what they mean in a main.py run and the two are comparable. It
never touches the loss or the gradients.)

The objective is Eq. (soft_surprise) of the appendix. With softmax assignments at
temperature tau,

    pi_n  = softmax(y_n^(i) / tau),          rho_n = softmax(y_n^(j) / tau),

every count becomes a soft expectation,

    q_hat_k = (1/N)  sum_n pi_{n,k} rho_{n,k},
    p_k     = (1/2N) sum_n (pi_{n,k} + rho_{n,k}),
    q_k     = p_k^2,

the over-matching indicator 1[q_hat_k > q_k] becomes a logistic gate of width eps, and

    S_soft(theta; tau, eps) = sum_k sigmoid((q_hat_k - q_k) / eps) * D(q_hat_k || q_k),
    D(a || b) = a log(a/b) + (1-a) log((1-a)/(1-b)).

Training minimizes -S_soft. Checkpoints are written in the same format as main.py
(full model objects), so `test.py` evaluates them with no changes.

To run on MNIST:
CUDA_VISIBLE_DEVICES=x python main_softmax.py \
    --save_dir ./model_bin/my_softmax_run/ \
    --dataset_name MNIST \
    --num_epochs 300
"""

import argparse
import gc
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scripts.deep_network import ResNet9
from scripts.pair_maker_CIFAR10 import PairMaker
from scripts.surprise_score import SurpriseScore   # logging/diagnostics ONLY


# ──────────────────────────────────────────────────────────────────────────────
# The differentiable objective
# ──────────────────────────────────────────────────────────────────────────────
def soft_surprise_score(logits_i, logits_j, tau, eps_gate, eps_num=1e-12):
    """Eq. (soft_surprise): S_soft(theta; tau, eps), differentiable in theta.

    Args:
        logits_i, logits_j : (B, K) raw network outputs on the two complementary views.
        tau                : softmax temperature. Note the ResNet9 head L2-normalizes its
                             output, so a logit vector has unit norm and its entries are
                             O(1/sqrt(K)); tau must be correspondingly small for the
                             softmax to be peaked at all (see --tau default).
        eps_gate           : width of the logistic gate. The margin q_hat_k - q_k is a
                             difference of probabilities, so it is O(1e-3); eps_gate must
                             be on that scale or the gate sits at 1/2 and stops selecting.
        eps_num            : clamp keeping the logarithms finite.

    Returns:
        (S_soft, q_hat, p, D, gate) -- S_soft is a scalar tensor with grad; the rest are
        detached-friendly (K,) tensors used for logging.

    !!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!
    Here is a line by line explanation of this function:
    
    Input is two (B, K) tensors; output is a scalar. Everything in between collapses the batch axis, so all intermediates are (K,) — one number per cluster.

    pi = F.softmax(logits_i / tau, dim=-1) → (B, K)
    Turns each image's raw logit vector into a probability distribution over the K clusters. dim=-1 means the softmax runs across clusters, so each row sums to 1. This is the soft stand-in for argmax: instead of "image n is in cluster 3", you get "image n is 88% cluster 3, 12% cluster 2". tau controls the sharpness — smaller tau → closer to one-hot → closer to the hard score. Same for rho on the other view. Shape is unchanged: (B,K) → (B,K).

    q_hat = (pi * rho).mean(dim=0) → (K,)
    Two steps. pi * rho is element-wise, staying (B,K): entry [n,k] = (view i's probability of cluster k) × (view j's probability of cluster k) = the soft probability that both views of image n chose k. Then .mean(dim=0) averages over n, collapsing the image axis — that mean is the 1/N Σ_n in the paper. Result: for each cluster k, how often the two views actually agreed on it.

    p = 0.5 * (pi.mean(dim=0) + rho.mean(dim=0)) → (K,)
    How often cluster k gets used at all, pooling both views. pi.mean(0) is view i's marginal, rho.mean(0) is view j's, and the 0.5 averages them — this is the 1/2N Σ_n (π + ρ) of eq:marg_5, and it's why the paper's p_k pools both views. p sums to 1.

    q = p ** 2 → (K,)
    The null: if the two views were independent, both landing on cluster k would happen with probability p_k × p_k. This is the baseline q̂_k must beat.

    D = q_hat*log(q_hat/q) + (1-q_hat)*log((1-q_hat)/(1-q)) → (K,)
    Binary KL between observed agreement and the independence baseline — per-cluster surprise. The two 1s are the "everything other than cluster k" outcome, which is what the appendix expands into the off-diagonal sum. The _c clamps only stop log(0).

    gate = torch.sigmoid((q_hat - q) / eps_gate) → (K,)
    The smooth replacement for 1[q̂_k > q_k]. Positive margin → ~1 (cluster counts), negative → ~0 (dropped). eps_gate is the ramp width.

    (gate * D).sum() → scalar
    Element-wise weight, then sum over clusters. One number, differentiable, and -S_soft is the loss.
    """
    pi = F.softmax(logits_i / tau, dim=-1)     # (B, K)
    rho = F.softmax(logits_j / tau, dim=-1)    # (B, K)

    # Soft counts. mean over the batch dim IS the 1/N (and 1/2N) of the appendix.
    q_hat = (pi * rho).mean(dim=0)                       # (K,)  = (1/N)  sum_n pi rho
    p = 0.5 * (pi.mean(dim=0) + rho.mean(dim=0))         # (K,)  = (1/2N) sum_n (pi + rho)
    q = p ** 2                                           # (K,)

    q_hat_c = q_hat.clamp(min=eps_num, max=1.0 - eps_num)
    q_c = q.clamp(min=eps_num, max=1.0 - eps_num)

    # Binary KL, D(q_hat_k || q_k)
    D = (q_hat_c * torch.log(q_hat_c / q_c)
         + (1.0 - q_hat_c) * torch.log((1.0 - q_hat_c) / (1.0 - q_c)))

    # Smooth gate replacing 1[q_hat_k > q_k]
    gate = torch.sigmoid((q_hat - q) / eps_gate)

    return (gate * D).sum(), q_hat, p, D, gate


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics: the HARD score, so train.log is comparable with a main.py run
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def hard_diagnostics(model, tensor_i, tensor_j, K):
    """Hard (argmax) surprise score / agreement / active clusters, for logging only.

    Uses exactly the statistic main.py logs, so a softmax run and an ES run can be read
    off the same axis. Evaluated in eval mode (running-stat BatchNorm), matching test.py.
    """
    was_training = model.training
    model.eval()
    seq_i = model(tensor_i).argmax(dim=-1).cpu().numpy()
    seq_j = model(tensor_j).argmax(dim=-1).cpu().numpy()
    if was_training:
        model.train()

    scorer = SurpriseScore(seq_i, seq_j, K)
    return {
        "per_dim_kl": float(scorer.score("per_dim_kl")),
        "kl": float(scorer.score("kl")),
        "valid_dims": float(scorer.score("valid_dims")),
        "agreement": float(scorer.score("view_agreement")),
        "active_i": int(len(np.unique(seq_i))),
    }


class SoftmaxOptimizer:
    def __init__(self, K=64, N=3000, base_width=8, dataset_name="MNIST",
                 save_dir="./model_bin/my_softmax_run/", num_epochs=300,
                 tau=0.02, eps_gate=1e-3, learning_rate=1e-3, weight_decay=0.0,
                 grad_clip=5.0, seed=None):

        self.K = K
        self.N = N
        self.base_width = base_width
        self.dataset_name = dataset_name
        self.save_dir = save_dir
        self.num_epochs = num_epochs
        self.tau = tau
        self.eps_gate = eps_gate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # SAME data pipeline as main.py -- identical masking and augmentation.
        self.pair_maker = PairMaker(dataset_name=self.dataset_name, batch_size=self.N)

        if self.dataset_name in ("MNIST", "FashionMNIST", "USPS"):
            self.in_channels = 1
        elif self.dataset_name == "CIFAR10":
            if self.pair_maker.use_sobel:
                self.in_channels = 2
            else:
                self.in_channels = 1 if self.pair_maker.to_grayscale else 3
        else:
            raise ValueError("Unsupported dataset. Use 'MNIST', 'FashionMNIST', 'USPS' or 'CIFAR10'.")

        self.model = self._build_network()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.weight_decay)

        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = logging.getLogger(self.save_dir)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.FileHandler(os.path.join(self.save_dir, "train.log"), mode="a")
            fh.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(fh)

    def _build_network(self):
        """Same ResNet9 as main.py: only the optimizer and the objective differ."""
        return ResNet9(K=self.K, in_channels=self.in_channels,
                       base_width=self.base_width, normalize=True).to(self.device)

    def _check_and_reinit(self, max_tries=500):
        """Re-initialize until the argmax is not collapsed, as main.py does.

        Kept so that the softmax run starts from the same kind of initialization as the
        ES run; a collapsed start would otherwise confound the comparison.
        """
        batch = next(iter(self.pair_maker.split_patches_from_loader(
            split="local_search", device=self.device)))
        tensor_i = batch[0]

        unique_dims = -1
        for attempt in range(max_tries):
            # TRAIN mode (batch-stat BatchNorm), exactly as main.py does this check. In
            # eval mode a freshly built net still carries BatchNorm's default running
            # stats (mean 0, var 1), which do not match its activations at all, so the
            # argmax collapses to a couple of dimensions and the check fails every time
            # however healthy the initialization actually is.
            self.model.train()
            with torch.no_grad():
                out = self.model(tensor_i)
            unique_dims = out.argmax(dim=-1).unique().numel()
            if unique_dims > self.K * 0.15:
                print(f"Good init found after {attempt+1} tries ({unique_dims} unique dims)")
                return
            print(f"Collapsed init (only {unique_dims} unique dims), re-initializing...")
            self.model = self._build_network()
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)

        print(f"!!! WARNING: no good init after {max_tries} tries "
              f"({unique_dims} unique dims). Results from this experiment are suspect.")
        self.logger.warning(f"WARNING: no good init after {max_tries} tries "
                            f"({unique_dims} unique dims) -- starting collapsed.")

    def optimize(self):
        self.model = self.model.to(self.device)
        self._check_and_reinit()

        print(f"[softmax] tau={self.tau}  eps_gate={self.eps_gate}  lr={self.learning_rate}  "
              f"K={self.K}  N={self.N}")
        self.logger.info(f"# softmax run | tau={self.tau} eps_gate={self.eps_gate} "
                         f"lr={self.learning_rate} K={self.K} N={self.N} "
                         f"base_width={self.base_width} dataset={self.dataset_name}")

        for epoch in range(self.num_epochs):
            self.model.train()
            running, n_steps = 0.0, 0
            tensor_i = tensor_j = None

            with tqdm(self.pair_maker.split_patches_from_loader(
                        split="local_search", device=self.device),
                      desc=f"epoch {epoch}", unit="batch") as pbar:

                for tensor_i, tensor_j, _, _ in pbar:
                    logits_i = self.model(tensor_i)
                    logits_j = self.model(tensor_j)

                    s_soft, _, _, _, _ = soft_surprise_score(
                        logits_i, logits_j, tau=self.tau, eps_gate=self.eps_gate)

                    loss = -s_soft                      # maximize the soft surprise score

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.grad_clip is not None and self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                    running += float(s_soft.detach())
                    n_steps += 1
                    pbar.set_postfix({"S_soft": f"{float(s_soft.detach()):.4f}"})

            mean_soft = running / max(n_steps, 1)

            # Hard-score diagnostics on the last batch, exactly as main.py logs them.
            diag = hard_diagnostics(self.model, tensor_i, tensor_j, self.K)

            print(f"Epoch {epoch} | S_soft: {mean_soft:.4f} | hard score: "
                  f"{diag['per_dim_kl']:.4f}, valid: {diag['valid_dims']}, "
                  f"agreement: {diag['agreement']:.4f}, active: {diag['active_i']}")
            self.logger.info(
                f"epoch {epoch} | "
                f"soft_score: {mean_soft:.4f}, "
                f"final_opt: {diag['per_dim_kl']:.4f}, "
                f"score_opt: {diag['kl']:.4f}, "
                f"valid_opt: {diag['valid_dims']}, "
                f"agreement_opt: {diag['agreement']:.4f}, "
                f"active_clusters: {diag['active_i']}"
            )

            # Same checkpoint convention as main.py, so test.py needs no changes.
            torch.save(self.model, f"{self.save_dir}optimal_autoencoder.pth")
            if epoch > 0 and (epoch + 1) % 100 == 0:
                torch.save(self.model, f"{self.save_dir}optimal_autoencoder_epoch_{epoch+1}.pth")

            meta = {
                "objective": "soft_surprise (end-to-end Adam)",
                "K": self.K, "N": self.N, "base_width": self.base_width,
                "dataset_name": self.dataset_name, "save_dir": self.save_dir,
                "tau": self.tau, "eps_gate": self.eps_gate,
                "learning_rate": self.learning_rate, "weight_decay": self.weight_decay,
                "num_epochs": self.num_epochs, "total_epochs_trained": epoch + 1,
            }
            with open(os.path.join(self.save_dir, "meta_info.json"), "w") as f:
                json.dump(meta, f, indent=4)

            torch.cuda.empty_cache()

        self.model = self.model.to("cpu")
        print("Optimization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train by maximizing the SOFT (differentiable) surprise score, end to end.")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)

    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--N", type=int, default=3000, help="Batch size; the score is a batch statistic")
    parser.add_argument("--base_width", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=300)

    parser.add_argument("--tau", type=float, default=0.02,
                        help="Softmax temperature. The ResNet9 head L2-normalizes its output, "
                             "so entries are O(1/sqrt(K)) and tau must be small.")
    parser.add_argument("--eps_gate", type=float, default=1e-3,
                        help="Width of the logistic gate on (q_hat_k - q_k).")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--number_of_experiments", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    for experiment in range(args.number_of_experiments):
        current_save_dir = args.save_dir + str(experiment) + "/"
        opt = SoftmaxOptimizer(
            K=args.K, N=args.N, base_width=args.base_width,
            dataset_name=args.dataset_name, save_dir=current_save_dir,
            num_epochs=args.num_epochs, tau=args.tau, eps_gate=args.eps_gate,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            seed=None if args.seed is None else args.seed + experiment,
        )
        opt.optimize()

        del opt
        gc.collect()
        torch.cuda.empty_cache()

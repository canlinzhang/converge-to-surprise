"""
End-to-end differentiable baseline #2: maximize the GUMBEL-softmax surprise score.

Companion to `main_softmax.py`. Same data, same network, same objective shape -- the
only change is the relaxation used in place of the hard argmax:

    main.py           hard S(theta), argmax counts        -> gradient-free ES + surrogate FT
    main_softmax.py   pi = softmax(z / tau)               -> Adam, end to end
    main_softmax_2.py pi = softmax((z + g) / tau)         -> Adam, end to end   <-- this file
                      g ~ Gumbel(0,1) i.i.d. per (image, cluster)

This implements Section "The Gumbel-Softmax relaxation"

    pi_n  = softmax((y_n^(i) + g_n^(i)) / tau),   rho_n = softmax((y_n^(j) + g_n^(j)) / tau),
    q_hat_k = (1/N)  sum_n pi_{n,k} rho_{n,k},
    p_k     = (1/2N) sum_n (pi_{n,k} + rho_{n,k}),
    q_k     = p_k^2,
    S_soft  = sum_k sigmoid((q_hat_k - q_k) / eps) * D(q_hat_k || q_k),

with the Gumbel noise drawn independently for the two views and re-drawn every step.
Dropping g recovers `main_softmax.py` exactly, which is the note's own remark.

Nothing under ./scripts/ is modified, and K / N / the augmentation come from the same
`PairMaker` and the same ResNet9 as every other run, so the arms stay comparable.
(SurpriseScore is imported for LOGGING only, as in main_softmax.py.)

NOTE ON THE NOISE SCALE (read before interpreting results). Gumbel(0,1) has std ~1.28,
whereas the ResNet9 head L2-normalizes its output, so the logits z have std ~0.12 and a
mean top-2 gap of ~0.018. Taken literally, z + g is therefore ~10x noise to signal: the
category being sampled comes from softmax(z), which for such small z is nearly uniform.
`--gumbel_scale` exists to control this. At the default 1.0 the file implements the note
verbatim; setting it to tau (i.e. --gumbel_scale 0.02 with --tau 0.02) instead samples
from softmax(z / tau), the same categorical `main_softmax.py` uses, which is the reading
under which the Gumbel trick does what it is normally understood to do.

To run on MNIST:
CUDA_VISIBLE_DEVICES=x python main_softmax_2.py \
    --save_dir ./model_bin/my_gumbel_run/ --dataset_name MNIST --num_epochs 300
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
# Gumbel-softmax sampling
# ──────────────────────────────────────────────────────────────────────────────
def sample_gumbel(shape, device, dtype, eps=1e-20):
    """i.i.d. Gumbel(0,1) via inverse CDF: g = -log(-log(u)), u ~ Uniform(0,1).

    The two eps guards keep the logs finite when u underflows to 0 or rounds to 1.
    """
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_softmax_assignments(logits, tau, gumbel_scale=1.0, straight_through=False):
    """One Gumbel-softmax sample per (image, cluster): softmax((z + s*g) / tau).

    Gradients reach `logits` through the reparameterization -- the noise is additive and
    parameter-free, so d/dz is the ordinary softmax Jacobian.

    straight_through is NOT part of the note. When enabled, the forward value is the
    one-hot argmax while the backward pass uses the soft Jacobian; the surprise score is
    then computed on genuinely discrete assignments. Off by default.
    """
    g = sample_gumbel(logits.shape, logits.device, logits.dtype)
    y = F.softmax((logits + gumbel_scale * g) / tau, dim=-1)
    if straight_through:
        idx = y.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
        y = (y_hard - y).detach() + y          # forward: hard, backward: soft
    return y


def gumbel_surprise_score(logits_i, logits_j, tau, eps_gate, gumbel_scale=1.0,
                          straight_through=False, eps_num=1e-12):
    """S_soft under the Gumbel-softmax relaxation. Differentiable in theta.

    Identical to `main_softmax.soft_surprise_score` downstream of the assignments; only
    pi/rho differ. Independent noise per view, re-drawn on every call.

    Args:
        logits_i, logits_j : (B, K) raw network outputs on the two complementary views.
    Returns:
        (S_soft, q_hat, p, D, gate); S_soft is a scalar with grad, the rest are (K,).
    """
    pi = gumbel_softmax_assignments(logits_i, tau, gumbel_scale, straight_through)
    rho = gumbel_softmax_assignments(logits_j, tau, gumbel_scale, straight_through)

    q_hat = (pi * rho).mean(dim=0)                       # (K,)  = (1/N)  sum_n pi rho
    p = 0.5 * (pi.mean(dim=0) + rho.mean(dim=0))         # (K,)  = (1/2N) sum_n (pi + rho)
    q = p ** 2                                           # (K,)

    q_hat_c = q_hat.clamp(min=eps_num, max=1.0 - eps_num)
    q_c = q.clamp(min=eps_num, max=1.0 - eps_num)

    D = (q_hat_c * torch.log(q_hat_c / q_c)
         + (1.0 - q_hat_c) * torch.log((1.0 - q_hat_c) / (1.0 - q_c)))

    gate = torch.sigmoid((q_hat - q) / eps_gate)

    return (gate * D).sum(), q_hat, p, D, gate


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics: the HARD score, so train.log is comparable across all arms
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def hard_diagnostics(model, tensor_i, tensor_j, K):
    """Hard (argmax) surprise score / agreement / active clusters, for logging only.

    No Gumbel noise here: the deployed model is the deterministic argmax of the raw
    logits, which is what test.py evaluates. Eval mode (running-stat BatchNorm), also
    matching test.py.
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


class GumbelOptimizer:
    def __init__(self, K=64, N=3000, base_width=8, dataset_name="MNIST",
                 save_dir="./model_bin/my_gumbel_run/", num_epochs=300,
                 tau=0.02, eps_gate=1e-3, gumbel_scale=1.0, straight_through=False,
                 learning_rate=1e-3, weight_decay=0.0, grad_clip=5.0, seed=None):

        self.K = K
        self.N = N
        self.base_width = base_width
        self.dataset_name = dataset_name
        self.save_dir = save_dir
        self.num_epochs = num_epochs
        self.tau = tau
        self.eps_gate = eps_gate
        self.gumbel_scale = gumbel_scale
        self.straight_through = straight_through
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # SAME data pipeline as main.py / main_softmax.py.
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
        """Same ResNet9 as every other arm."""
        return ResNet9(K=self.K, in_channels=self.in_channels,
                       base_width=self.base_width, normalize=True).to(self.device)

    def _check_and_reinit(self, max_tries=500):
        """Re-initialize until the argmax is not collapsed, as main.py does.

        TRAIN mode (batch-stat BatchNorm): in eval mode a fresh net still carries
        BatchNorm's default running stats, so the argmax collapses however healthy the
        initialization actually is.
        """
        batch = next(iter(self.pair_maker.split_patches_from_loader(
            split="local_search", device=self.device)))
        tensor_i = batch[0]

        unique_dims = -1
        for attempt in range(max_tries):
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

        print(f"[gumbel] tau={self.tau}  eps_gate={self.eps_gate}  "
              f"gumbel_scale={self.gumbel_scale}  straight_through={self.straight_through}  "
              f"lr={self.learning_rate}  K={self.K}  N={self.N}")
        self.logger.info(f"# gumbel run | tau={self.tau} eps_gate={self.eps_gate} "
                         f"gumbel_scale={self.gumbel_scale} "
                         f"straight_through={self.straight_through} "
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

                    s_soft, _, _, _, _ = gumbel_surprise_score(
                        logits_i, logits_j, tau=self.tau, eps_gate=self.eps_gate,
                        gumbel_scale=self.gumbel_scale,
                        straight_through=self.straight_through)

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
                "objective": "gumbel_softmax_surprise (end-to-end Adam)",
                "K": self.K, "N": self.N, "base_width": self.base_width,
                "dataset_name": self.dataset_name, "save_dir": self.save_dir,
                "tau": self.tau, "eps_gate": self.eps_gate,
                "gumbel_scale": self.gumbel_scale,
                "straight_through": self.straight_through,
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
        description="Train by maximizing the GUMBEL-softmax surprise score, end to end.")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)

    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--N", type=int, default=3000, help="Batch size; the score is a batch statistic")
    parser.add_argument("--base_width", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=300)

    parser.add_argument("--tau", type=float, default=0.02,
                        help="Relaxation temperature. The ResNet9 head L2-normalizes its "
                             "output, so entries are O(1/sqrt(K)) and tau must be small.")
    parser.add_argument("--eps_gate", type=float, default=1e-3,
                        help="Width of the logistic gate on (q_hat_k - q_k).")
    parser.add_argument("--gumbel_scale", type=float, default=1.0,
                        help="Multiplier on the Gumbel noise. 1.0 implements the note "
                             "verbatim, softmax((z + g)/tau). Because Gumbel(0,1) has std "
                             "~1.28 while the L2-normalized logits have std ~0.12, that is "
                             "~10x noise to signal; passing the same value as --tau instead "
                             "samples from softmax(z/tau), the categorical main_softmax.py uses.")
    parser.add_argument("--straight_through", action="store_true",
                        help="Straight-through estimator: hard one-hot forward, soft backward. "
                             "Not part of the note; off by default.")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--number_of_experiments", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    for experiment in range(args.number_of_experiments):
        current_save_dir = args.save_dir + str(experiment) + "/"
        opt = GumbelOptimizer(
            K=args.K, N=args.N, base_width=args.base_width,
            dataset_name=args.dataset_name, save_dir=current_save_dir,
            num_epochs=args.num_epochs, tau=args.tau, eps_gate=args.eps_gate,
            gumbel_scale=args.gumbel_scale, straight_through=args.straight_through,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            seed=None if args.seed is None else args.seed + experiment,
        )
        opt.optimize()

        del opt
        gc.collect()
        torch.cuda.empty_cache()

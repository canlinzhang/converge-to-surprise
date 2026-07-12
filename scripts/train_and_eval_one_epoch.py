import os
import torch
import json
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from scripts.surprise_score import SurpriseScore


def train_and_eval_one_epoch(
    epoch,
    autoencoder,
    pair_maker,
    optimizer,
    criterion,
    device,
    K,
    kl_threshold=0.005,
    train_batch_size=128,
    num_train_epochs=3
):
    """
    One epoch of:
      1. Accumulate contributing positions across all batches (inference pass)
      2. Build a supervised training set from contributing (image, pseudo_label) pairs
      3. Train the autoencoder on new _make_pairs augmentations of those images

    Args:
        epoch        (int)         : current epoch index (for logging)
        autoencoder  (nn.Module)   : the deep network, maps (B,C,H,W) -> (B,K)
        pair_maker   (PairMaker)   : data pipeline
        optimizer                  : torch optimizer
        criterion                  : loss function, e.g. nn.CrossEntropyLoss()
        device       (torch.device): cuda or cpu
        K            (int)         : number of output dimensions / alphabet size
        kl_threshold (float)       : per-dimension KL threshold for contributing_positions
        train_batch_size   (int)         : mini-batch size for the training phase
        num_train_epochs   (int)         : number of training epochs for supervised fine-tuning

    Returns:
        dict with keys: n_contributing, n_total, train_loss
    """

    for _ in range(num_train_epochs):

        autoencoder.eval()

        #modify the train data in each train epoch.
        #pair_maker.make_session_loaders()

        # ── Phase 1: Accumulate holders across all batches ───────────────────────
        image_holder   = []   # original images (CPU tensors)
        seq_i_holder   = []   # argmax sequences from tensor_i
        seq_j_holder   = []   # argmax sequences from tensor_j

        with torch.no_grad():
            with tqdm(pair_maker.split_patches_from_loader(split='local_search', device=device, force_augment=0),
                    desc=f"epoch {epoch} [accumulate]", unit="batch") as pbar:

                for tensor_i, tensor_j, _, images in pbar:
                    # Forward pass → embeddings → argmax sequences
                    emb_i = autoencoder(tensor_i)          # (B, K)
                    emb_j = autoencoder(tensor_j)          # (B, K)
                    seq_i = emb_i.argmax(dim=-1).cpu().tolist()
                    seq_j = emb_j.argmax(dim=-1).cpu().tolist()

                    # Accumulate — keep images on CPU to save GPU memory
                    image_holder.extend(images.cpu())      # list of (C,H,W) tensors
                    seq_i_holder.extend(seq_i)             # list of ints
                    seq_j_holder.extend(seq_j)             # list of ints

        # ── Phase 2: Filter contributing positions ───────────────────────────────
        scorer   = SurpriseScore(seq_i_holder, seq_j_holder, K=K)
        mask_arr = scorer.contributing_positions(method="kl", threshold=kl_threshold)  # (N_total,)
        kl_score_per_dim = scorer.score("per_dim_kl")
        contrib_idx = np.where(mask_arr == 1)[0]

        n_total        = len(mask_arr)
        n_contributing = len(contrib_idx)
        print(f"  Contributing positions: {n_contributing} / {n_total} "
            f"({100 * n_contributing / max(n_total, 1):.1f}%)")

        if n_contributing == 0:
            print("  No contributing positions found — skipping training phase.")
            return {"n_contributing": 0, "n_total": n_total, "train_loss": float("nan")}

        # ── Phase 2b: Median-based balancing across dimensions ───────────────────
        # Group contributing indices by their pseudo-label (dimension k)
        seq_i_arr = np.array(seq_i_holder)
        dim_to_indices = {}
        for idx in contrib_idx:
            k = seq_i_arr[idx]
            if k not in dim_to_indices:
                dim_to_indices[k] = []
            dim_to_indices[k].append(idx)

        # Compute median occurrence count across all contributing dimensions
        counts_per_dim = np.array([len(v) for v in dim_to_indices.values()])
        median_count = int(np.median(counts_per_dim))
        print(f"  Dimensions contributing: {len(dim_to_indices)} | "
            f"median count: {median_count} | "
            f"counts: { {k: len(v) for k, v in sorted(dim_to_indices.items())} }")

        # For each dimension, cap at median (random sample if over, take all if under)
        balanced_idx = []
        for k, indices in dim_to_indices.items():
            indices = np.array(indices)
            if len(indices) >= median_count:
                chosen = np.random.choice(indices, size=median_count, replace=False)
            else:
                chosen = indices  # take all
            balanced_idx.append(chosen)

        balanced_idx = np.concatenate(balanced_idx)  # final selected indices

        # Collect contributing images and pseudo-labels
        # pseudo_label[n] = seq_i_holder[n]  (== seq_j_holder[n] at match positions)
        contrib_images = torch.stack([image_holder[n] for n in balanced_idx])   # (M, C, H, W)
        contrib_labels = torch.tensor(
            [seq_i_arr[n] for n in balanced_idx], dtype=torch.long
        )                                                                         # (M,)

        n_select = len(balanced_idx)
        print(f"  After balancing: {n_select} images selected")

        # ── Phase 3: Train on contributing images ────────────────────────────────
        autoencoder.train()
        total_loss   = 0.0
        n_batches    = 0

        # Shuffle balanced set
        perm = torch.randperm(n_select)
        contrib_images = contrib_images[perm]          # (n_select, C, H, W)
        contrib_labels = contrib_labels[perm]          # (n_select,)

        for start in range(0, n_select, train_batch_size):  # noqa
            end      = min(start + train_batch_size, n_select)
            imgs_b   = contrib_images[start:end]    # (b, C, H, W)
            labels_b = contrib_labels[start:end].to(device)

            # Fresh disjoint-pixel pairs + augmentation via _make_pairs
            tensor_i, tensor_j = pair_maker._make_pairs(imgs_b, device=device)

            optimizer.zero_grad()

            # Forward on both views, supervised by pseudo-label
            logits_i = autoencoder(tensor_i)        # (b, K)
            logits_j = autoencoder(tensor_j)        # (b, K)

            loss = criterion(logits_i, labels_b) + criterion(logits_j, labels_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Train loss: {avg_loss:.4f}")

    return {
        "n_contributing": n_contributing,
        "n_total":        n_total,
        "train_loss":     avg_loss,
    }


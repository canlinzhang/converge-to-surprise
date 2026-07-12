import os
import copy
import argparse
import numpy as np
import torch
from tqdm import tqdm

from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI

from scripts.surprise_score import SurpriseScore
from scripts.pair_maker import PairMaker   # adjust import to match your project
from scripts.utils import cluster_acc, best_cluster_fit

'''
To run
CUDA_VISIBLE_DEVICES=x python test.py \
    --save_dir ./model_bin/xxxx/ \
    --dataset_name MNIST or FashionMNIST or USPS
'''

# ──────────────────────────────────────────────
# Cluster → Label assignment helpers
# ──────────────────────────────────────────────

def build_confusion_matrix(all_preds, all_labels, num_clusters, num_classes):
    """
    Build a (num_clusters x num_classes) confusion matrix.
    C[i, j] = number of samples assigned to cluster i with true label j.
    """
    C = np.zeros((num_clusters, num_classes), dtype=np.int64)
    for pred, label in zip(all_preds, all_labels):
        C[pred, label] += 1
    return C


def cluster_purity_report(C, num_classes, top_n=None):
    """
    Print the label composition (purity) of each cluster.

    Args:
        C           : (num_clusters x num_classes) confusion matrix from build_confusion_matrix()
        num_classes : number of ground-truth classes
        top_n       : if set, only print the top_n most populated clusters (sorted by size).
                      If None, print all clusters including dead ones.

    For each cluster k, prints:
      - total samples assigned
      - purity  = count_of_majority_class / total
      - dominant class
      - full label breakdown: count and % for each class
    """
    num_clusters = C.shape[0]
    cluster_sizes = C.sum(axis=1)          # (K,)

    # Sort by cluster size descending
    order = np.argsort(-cluster_sizes)
    if top_n is not None:
        order = order[:top_n]

    print(f"\n{'='*60}")
    print(f"  Cluster Purity Report  (K={num_clusters}, classes={num_classes})")
    print(f"{'='*60}")

    for k in order:
        total = cluster_sizes[k]
        if total == 0:
            print(f"  Cluster {k:3d} | DEAD (0 samples)")
            continue

        dominant_class = int(np.argmax(C[k]))
        purity = C[k, dominant_class] / total

        # Build compact label breakdown string: only show classes with > 0 samples
        breakdown = "  ".join(
            f"cls{cls}:{C[k, cls]}({100*C[k,cls]/total:.0f}%)"
            for cls in range(num_classes)
            if C[k, cls] > 0
        )

        print(
            f"  Cluster {k:3d} | n={int(total):5d} | "
            f"purity={purity*100:.1f}% (dom=cls{dominant_class}) | {breakdown}"
        )

    # Summary statistics
    active_mask = cluster_sizes > 0
    if active_mask.sum() > 0:
        purities = np.array([
            C[k, np.argmax(C[k])] / cluster_sizes[k]
            for k in range(num_clusters) if cluster_sizes[k] > 0
        ])
        print(f"\n  Active clusters : {active_mask.sum()} / {num_clusters}")
        print(f"  Mean purity     : {purities.mean()*100:.2f}%")
        print(f"  Min  purity     : {purities.min()*100:.2f}%")
        print(f"  Max  purity     : {purities.max()*100:.2f}%")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────
# Main tester
# ──────────────────────────────────────────────

class Tester:
    def __init__(self, args):
        self.dataset_name   = args.dataset_name
        self.save_dir       = args.save_dir
        self.K              = args.K
        self.N              = args.batch_size
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Number of ground-truth classes
        self.num_classes = {
            "MNIST":        10,
            "FashionMNIST": 10,
            "USPS":         10,
            "CIFAR10":      10,
            "CIFAR100":     100,
        }.get(self.dataset_name, 10)

        # Load the evolved model
        model_path = os.path.join(self.save_dir, "optimal_autoencoder.pth")
        print(f"Loading model from {model_path}")
        self.autoencoder = torch.load(model_path, map_location=self.device)
        self.autoencoder.eval()

        self.pair_maker = PairMaker(dataset_name=self.dataset_name, batch_size=self.N)

    @torch.no_grad()
    def run(self):
        all_preds  = []
        all_preds_j = []
        all_labels = []

        # ── Collect predictions over the entire test set ──
        with tqdm(
            self.pair_maker.split_patches_from_loader(split="test", device=self.device),
            desc="Testing", unit="batch"
        ) as pbar:
            for tensor_i, tensor_j, labels, _ in pbar:
                # Use only one view (tensor_i) for final cluster assignment
                emb_i  = self.autoencoder(tensor_i)
                emb_j  = self.autoencoder(tensor_j)
                preds  = emb_i.argmax(dim=-1).cpu().numpy()   # (B,) cluster indices from view i
                preds_j = emb_j.argmax(dim=-1).cpu().numpy()  # (B,) cluster indices from view j

                # labels may be a tensor or list
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                else:
                    labels = np.array(labels)

                all_preds.append(preds)
                all_preds_j.append(preds_j)
                all_labels.append(labels)

                pbar.set_postfix(batches=len(all_preds))

        all_preds  = np.concatenate(all_preds,   axis=0)   # (N_test,)
        all_preds_j = np.concatenate(all_preds_j, axis=0)  # (N_test,)
        all_labels = np.concatenate(all_labels,  axis=0)   # (N_test,)

        print(f"\nTotal test samples: {len(all_preds)}")
        print(f"Unique predicted clusters used: {len(np.unique(all_preds))} / {self.K}")

        # ── Build confusion matrix ──
        C = build_confusion_matrix(all_preds, all_labels, self.K, self.num_classes)

        # ── Evaluate: ACC / NMI / ARI (DeepDPM protocol) ──
        # cluster_acc uses DeepDPM's square-Hungarian assignment; NMI and ARI
        # are sklearn's, called with the same (pred, label) arg order and the
        # same np.round(., 5) as DeepDPM's DeepDPM.py.
        acc = np.round(cluster_acc(all_labels, all_preds), 5)
        nmi = np.round(NMI(all_preds, all_labels), 5)
        ari = np.round(ARI(all_preds, all_labels), 5)
        final_K = len(np.unique(all_preds))

        _, row_ind, col_ind, _ = best_cluster_fit(all_labels, all_preds)

        # Build mapping: predicted cluster k → assigned class (or -1 if paired
        # with a dummy column ≥ num_classes).
        mapping = {}
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            mapping[int(r)] = int(c) if c < self.num_classes else -1

        print(f"\n[DeepDPM Square-Hungarian Assignment]  (K={self.K}, classes={self.num_classes})")
        print(f"Cluster → Class mapping: {mapping}")
        print(f"\n{'='*40}")
        print(f"  ACC: {acc * 100:.2f}%   (acc={acc})")
        print(f"  NMI: {nmi}")
        print(f"  ARI: {ari}")
        print(f"  final K (clusters used): {final_K}")
        print(f"{'='*40}\n")

        # ── Per-class breakdown ──
        print("Per ground-truth class accuracy:")
        for cls in range(self.num_classes):
            cls_mask    = (all_labels == cls)
            cls_preds   = all_preds[cls_mask]
            # Map predictions to classes using the found mapping
            mapped_preds = np.array([mapping.get(int(p), -1) for p in cls_preds])
            cls_acc     = (mapped_preds == cls).mean() if cls_mask.sum() > 0 else 0.0
            print(f"  Class {cls:3d}: {cls_acc * 100:.1f}%  (n={cls_mask.sum()})")

        # ── Cluster purity report ──
        cluster_purity_report(C, self.num_classes)

        # ── Surprise score on test set ──
        seq_i = all_preds.tolist()
        seq_j = all_preds_j.tolist()
        scorer = SurpriseScore(seq_i[:3000], seq_j[:3000], self.K)
        #print(f"\nper_dim_kl surprise (view_i vs view_j, test set): {scorer.per_dim_kl_surprise():.4f}")

        print(f"View agreement: {(all_preds == all_preds_j).sum()} / {len(all_preds)} matched ({(all_preds == all_preds_j).mean()*100:.2f}%)")

        # DeepDPM-style one-line summary for easy copy into tables.
        print(f"\nNMI: {nmi}, ARI: {ari}, acc: {acc}, final K: {final_K}")
        return {"acc": acc, "nmi": nmi, "ari": ari, "final_K": final_K, "mapping": mapping}


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",      type=str,  required=True)
    parser.add_argument("--dataset_name",  type=str,  default="MNIST")
    parser.add_argument("--K",             type=int,  default=64)
    parser.add_argument("--batch_size",    type=int,  default=2000)
    args = parser.parse_args()

    tester = Tester(args)
    tester.run()

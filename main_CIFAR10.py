import os
import torch
import numpy as np
import shutil
import math
import random
import copy
import argparse
import gc
import json
import logging

from functools import reduce
from collections import defaultdict, Counter
from tqdm import tqdm

from scripts.deep_network import ResNet9
from scripts.pair_maker_CIFAR10 import PairMaker
from scripts.evolution_strategy import EvolutionStrategy
from scripts.evolution_strategy import eval_score
from scripts.train_and_eval_one_epoch import train_and_eval_one_epoch


'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Note: We use the name 'autoencoder' to refer to the model being optimized.
But 'autoencoder' can be any deep network, not necessarily an autoencoder.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

To run on MNIST:
CUDA_VISIBLE_DEVICES=x python main.py \
    --save_dir ./model_bin/my_run/ \
    --dataset_name MNIST \
    --train_start_epoch_1 2000 \
    --train_start_epoch_2 2000 \
    --num_epochs 3001

To run on USPS: 
CUDA_VISIBLE_DEVICES=x python main.py \
    --save_dir ./model_bin/my_run/ \
    --dataset_name USPS \
    --train_start_epoch_1 4000 \
    --train_start_epoch_2 8000 \
    --num_epochs 9001 \
    --N 3650

To run on FashionMNIST: (slightly enlarge the network to handle more complex data)
CUDA_VISIBLE_DEVICES=x python main.py \
    --save_dir ./model_bin/my_run/ \
    --dataset_name FashionMNIST \
    --train_start_epoch_1 2000 \
    --train_start_epoch_2 2000 \
    --num_epochs 3001

To run on CIFAR10: (ResNet9 base_width=8 — the SAME net as the digit datasets —
on per-view greyscale + Sobel views from scripts/pair_maker_CIFAR10.py; in_channels=2)
CUDA_VISIBLE_DEVICES=x python main_CIFAR10.py \
    --save_dir ./model_bin/my_run/ \
    --dataset_name CIFAR10 \
    --train_start_epoch_1 2000 \
    --train_start_epoch_2 2000 \
    --num_epochs 3001 \
    --N 3125
'''
class Optimizer:
    def __init__(self, K=64, num_para_min=10, num_para_max=2000, N=3000, sigma=0.02, beta=0.5, base_width=8, dataset_name='MNIST',
                 learning_rate = 0.005, population_size = 30, weight_decay = 0.005,
                 save_dir='./model_bin/my_run/', consider_active_dim=True, num_epochs=10000, kl_threshold=0.005, train_batch_size=128, num_train_epochs=5, 
                 train_start_epoch_1=2000, train_start_epoch_2=4000, in_between_es_epoch_1=500, in_between_es_epoch_2=25,
                 resume=False, init_from=None):
        
        self.K = K #output dimension of deep network
        self.num_para_min = num_para_min
        self.num_para_max = num_para_max
        self.N = N
        self.sigma = sigma
        self.beta = beta
        self.base_width = base_width
        self.dataset_name = dataset_name
        self.save_dir = save_dir
        self.learning_rate = learning_rate
        self.population_size = population_size
        self.weight_decay = weight_decay
        self.consider_active_dim = consider_active_dim
        self.num_epochs = num_epochs
        self.kl_threshold = kl_threshold
        self.train_batch_size = train_batch_size
        self.num_train_epochs = num_train_epochs
        self.train_start_epoch_1 = train_start_epoch_1
        self.train_start_epoch_2 = train_start_epoch_2
        self.in_between_es_epoch_1 = in_between_es_epoch_1
        self.in_between_es_epoch_2 = in_between_es_epoch_2
        self.resume = resume
        self.init_from = init_from

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build the data pipeline FIRST: for CIFAR10 the number of input channels
        # depends on whether PairMaker converts to grayscale (1 channel) or keeps
        # raw RGB (3), so the network cannot be built until that choice is known.
        self.pair_maker = PairMaker(dataset_name=self.dataset_name, batch_size=self.N)

        if self.dataset_name in ("MNIST", "FashionMNIST", "USPS"):
            self.in_channels = 1
        elif self.dataset_name == 'CIFAR10':
            # Channel count follows PairMaker's preprocessing, read from its flags
            # so the two stay in sync automatically:
            #   per-view Sobel -> 2 channels [grad-x, grad-y] (checked first, since
            #       Sobel runs after the grayscale step and overrides its width)
            #   grayscale only -> 1 luma channel
            #   neither        -> raw 3-channel RGB
            if self.pair_maker.use_sobel:
                self.in_channels = 2
            else:
                self.in_channels = 1 if self.pair_maker.to_grayscale else 3
        else:
            raise ValueError("Unsupported dataset. For now we only accept 'MNIST', 'FashionMNIST', 'USPS', and 'CIFAR10'.")

        # Instantiate once — shared weights for both views. Every dataset,
        # CIFAR10 included, uses the same ResNet9.
        self.optimal_autoencoder = self._build_network()
        
        # Resume from a previous run in save_dir, ONLY when --resume is passed.
        # Fresh runs (the default) keep start_epoch=0 and the freshly-built network,
        # so every existing workflow — MNIST/FashionMNIST/USPS and any non-resumed
        # run — behaves exactly as before (this whole block is skipped).
        self.start_epoch = 0

        # --init_from: start ES from the weights of a DIFFERENT, finished run (e.g. a
        # model trained end-to-end by main_softmax.py), rather than from a fresh random
        # init. This differs from --resume in three ways that matter:
        #   * the checkpoint is read from an arbitrary path, not from this run's own
        #     save_dir, so every experiment 0..N-1 can start from the SAME weights;
        #   * the epoch counter stays at 0, so the ES/fine-tuning schedule
        #     (train_start_epoch_1/2, num_epochs) is applied in full to the continued
        #     run instead of being shifted by however long the source run trained;
        #   * _check_and_reinit is skipped (see optimize), so the loaded weights are
        #     never silently discarded. That check re-initializes whenever fewer than
        #     15% of the K output dimensions are used, and a converged non-parametric
        #     model legitimately uses ~10 of 64 (15.6%) -- right at the threshold.
        self.externally_initialized = False
        if self.init_from:
            if self.resume:
                raise ValueError("--init_from and --resume are mutually exclusive.")
            loaded = torch.load(self.init_from, map_location=self.device, weights_only=False)
            state = loaded.state_dict() if hasattr(loaded, "state_dict") else loaded
            # load_state_dict validates the architecture for us.
            self.optimal_autoencoder.load_state_dict(state)
            self.externally_initialized = True
            print(f"[init_from] loaded weights from {self.init_from}; "
                  f"epoch counter starts at 0, init check skipped")

        if self.resume:
            ckpt_path = os.path.join(self.save_dir, 'optimal_autoencoder.pth')
            meta_path = os.path.join(self.save_dir, 'meta_info.json')
            if os.path.exists(ckpt_path):
                # Models are saved as full objects; load and copy weights into the
                # freshly-built net (load_state_dict also validates the architecture).
                loaded = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                self.optimal_autoencoder.load_state_dict(loaded.state_dict())
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        self.start_epoch = int(json.load(f).get('total_epochs_trained', 0))
                print(f"[resume] loaded {ckpt_path}; continuing from epoch "
                      f"{self.start_epoch} toward num_epochs={self.num_epochs}")
            else:
                print(f"[resume] no checkpoint at {ckpt_path}; starting fresh (epoch 0)")

        self.es = EvolutionStrategy(
            autoencoder_original = self.optimal_autoencoder,
            sigma                = self.sigma,
            learning_rate        = self.learning_rate,
            population_size      = self.population_size,   # must be even (mirrored sampling)
            K                    = self.K,
            beta                 = self.beta,   # your beta for valid_dims term
            weight_decay         = self.weight_decay,
            consider_active_dim  = self.consider_active_dim,
            device               = self.device,
        )
        
        #free GPU memory
        torch.cuda.empty_cache()

        # Setup logger
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = logging.getLogger(self.save_dir)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.FileHandler(os.path.join(self.save_dir, 'train.log'), mode='a')
            fh.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(fh)

        # Supervised fine-tuning optimizer and loss
        self.sup_criterion = torch.nn.CrossEntropyLoss()

    def _build_network(self):
        """Instantiate the deep network. ALL datasets — including CIFAR10 — use
        ResNet9 with the same base_width; only in_channels differs (2 for CIFAR10's
        Sobel views, 1 for MNIST/FashionMNIST/USPS). So CIFAR is the exact same
        network as the digit datasets."""
        return ResNet9(
            K=self.K, in_channels=self.in_channels,
            base_width=self.base_width, normalize=True
        ).to(self.device)

    def _check_and_reinit(self, max_tries=500):
        """Re-initialize until we get a non-collapsed starting point.

        How often a fresh init passes depends strongly on the input representation.
        Measured over 120 random inits on a 3125-image CIFAR batch: ~9% pass on
        1-channel grayscale views, but only ~1.7% on 2-channel Sobel views (Sobel
        maps are low-variance and concentrated near the zero-gradient level, so the
        argmax over 64 logits collapses far more readily). At 1.7% per try the old
        budget of 100 tries failed outright ~18% of the time — and this loop does
        not raise on exhaustion, so a run would silently proceed from a collapsed
        start. 500 tries drops that to ~0.02%; each try is a single forward pass of
        a 107k-parameter network, so the whole check still costs seconds. The digit
        datasets pass in one or two tries, so the larger budget never engages there.
        """
        dummy_batch = next(iter(self.pair_maker.split_patches_from_loader(
            split="local_search", device=self.device)))
        tensor_i = dummy_batch[0]

        unique_dims = -1
        for attempt in range(max_tries):
            with torch.no_grad():
                out = self.optimal_autoencoder(tensor_i)  # (N, K)
            argmaxes = out.argmax(dim=-1)
            unique_dims = argmaxes.unique().numel()
            if unique_dims > self.K * 0.15:  # at least 15% of dims used
                print(f"Good init found after {attempt+1} tries ({unique_dims} unique dims)")
                return
            # Re-initialize
            print(f"Collapsed init (only {unique_dims} unique dims), re-initializing...")
            self.optimal_autoencoder = self._build_network()
            self.es.autoencoder_original = self.optimal_autoencoder
            self.es.total_ids = sum(p.numel() for p in self.optimal_autoencoder.parameters())

        # Exhausted the budget: proceed, but say so loudly. Previously this fell
        # through silently, and the only trace was a missing "Good init" line.
        print(f"!!! WARNING: no good init after {max_tries} tries; starting from a "
              f"COLLAPSED network ({unique_dims} unique dims, need "
              f"> {self.K * 0.15:.1f}). Results from this experiment are suspect.")
        self.logger.warning(
            f"WARNING: no good init after {max_tries} tries "
            f"({unique_dims} unique dims) — starting collapsed."
        )

    def optimize(self):
        # Move optimal autoencoder to GPU before starting optimization
        self.optimal_autoencoder = self.optimal_autoencoder.to(self.device)

        # Only on a genuinely fresh random start: not on --resume, and not when the
        # weights came from --init_from (re-initializing those would silently throw the
        # loaded model away).
        if self.start_epoch == 0 and not self.externally_initialized:
            self._check_and_reinit()

        # Adaptive num_para setup
        current_num_para = self.num_para_min

        for epoch in range(self.start_epoch, self.num_epochs):

            current_num_para = min(int(self.es.total_ids * ((epoch+1) / 100)), self.es.total_ids, self.num_para_max)

            with tqdm(self.pair_maker.split_patches_from_loader(split="local_search", device=self.device),
                    desc=f'epoch {epoch}', unit='batch') as pbar:
                for tensor_i, tensor_j, _, _ in pbar:

                    autoencoder_temp = copy.deepcopy(self.optimal_autoencoder)

                    autoencoder_temp = self.es.run(
                        autoencoder = autoencoder_temp,
                        tensor_i    = tensor_i,
                        tensor_j    = tensor_j,
                        num_para    = current_num_para,
                    )

                    self.optimal_autoencoder.load_state_dict(autoencoder_temp.state_dict())

                    pbar.set_postfix({'status': 'optimizing...'})

            # eval to show result, using the last batch of tensors!!!!!
            final_opt, score_opt, valid_opt, agreement_opt = eval_score(
                self.optimal_autoencoder, tensor_i, tensor_j,
                self.K, self.beta, self.consider_active_dim
            )

            print(f"Epoch {epoch} | final score: {final_opt:.4f}, ES score: {score_opt:.4f}, valid: {valid_opt}, agreement: {agreement_opt:.4f}")

            stats_log = {
                "final_opt": final_opt,
                "score_opt": score_opt,
                "valid_opt": valid_opt,
                "agreement_opt": agreement_opt,
            }

            os.makedirs(self.save_dir, exist_ok=True)
            torch.save(self.optimal_autoencoder, f'{self.save_dir}optimal_autoencoder.pth')
            self.logger.info(
                f'epoch {epoch} | '
                f'final_opt: {stats_log["final_opt"]:.4f}, '
                f'score_opt: {stats_log["score_opt"]:.4f}, '
                f'valid_opt: {stats_log["valid_opt"]}, '
                f'agreement_opt: {stats_log["agreement_opt"]:.4f}'
            )
            torch.cuda.empty_cache()  # Free GPU memory if needed

            #train and evaluate one epoch of supervised fine-tuning on contributing positions
            if (self.train_start_epoch_1 <= epoch < self.train_start_epoch_2 and epoch % self.in_between_es_epoch_1 == 0) or (
                epoch >= self.train_start_epoch_2 and epoch % self.in_between_es_epoch_2 == 0):

                if self.train_start_epoch_1 <= epoch < self.train_start_epoch_2:
                    num_train_epochs = int(0.5 * self.num_train_epochs)
                if epoch >= self.train_start_epoch_2:
                    num_train_epochs = self.num_train_epochs

                autoencoder_temp = copy.deepcopy(self.optimal_autoencoder)
                sup_optimizer = torch.optim.Adam(autoencoder_temp.parameters(), lr=1e-3)

                sup_stats = train_and_eval_one_epoch(
                    epoch=epoch,
                    autoencoder=autoencoder_temp,
                    pair_maker=self.pair_maker,
                    optimizer=sup_optimizer,
                    criterion=self.sup_criterion,
                    device=self.device,
                    K=self.K,
                    kl_threshold=self.kl_threshold,
                    train_batch_size=self.train_batch_size,
                    num_train_epochs=num_train_epochs
                )

                self.optimal_autoencoder.load_state_dict(autoencoder_temp.state_dict())

                # eval to show result, using the last batch of tensors!!!!!
                final_opt_, score_opt_, valid_opt_, agreement_opt_ = eval_score(
                    self.optimal_autoencoder, tensor_i, tensor_j,
                    self.K, self.beta, self.consider_active_dim
                )

                stats_log = {
                    "final_opt": final_opt_,
                    "score_opt": score_opt_,
                    "valid_opt": valid_opt_,
                    "agreement_opt": agreement_opt_,
                }

                os.makedirs(self.save_dir, exist_ok=True)
                torch.save(self.optimal_autoencoder, f'{self.save_dir}optimal_autoencoder.pth')
                self.logger.info(
                    f'training one epoch {epoch} | '
                    f'final_opt: {stats_log["final_opt"]:.4f}, '
                    f'score_opt: {stats_log["score_opt"]:.4f}, '
                    f'valid_opt: {stats_log["valid_opt"]}, '
                    f'agreement_opt: {stats_log["agreement_opt"]:.4f}'
                )
                torch.cuda.empty_cache()

            #save checkpoint every T epochs
            if epoch > 0 and (epoch+1) % 1000 == 0:
                torch.save(self.optimal_autoencoder, f'{self.save_dir}optimal_autoencoder_epoch_{epoch+1}.pth')

            # Save meta_info.json
            meta_path = os.path.join(self.save_dir, 'meta_info.json')
            if os.path.exists(meta_path):
                os.remove(meta_path)
            meta = {
                'K': self.K,
                'num_para_min': self.num_para_min,
                'num_para_max': self.num_para_max,
                'N': self.N,
                'sigma': self.sigma,
                'beta': self.beta,
                'base_width': self.base_width,
                'dataset_name': self.dataset_name,
                'save_dir': self.save_dir,
                'learning_rate': self.learning_rate,
                'population_size': self.population_size,
                'weight_decay': self.weight_decay,
                'consider_active_dim': self.consider_active_dim,
                'num_epochs': self.num_epochs,
                'total_epochs_trained': epoch + 1,
                'init_from': self.init_from,
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=4)
            print(f'Saved meta_info.json (total_epochs_trained={meta["total_epochs_trained"]})')

        # Move the optimal autoencoder back to CPU after all epochs are complete
        self.optimal_autoencoder = self.optimal_autoencoder.to('cpu')

        print("Optimization complete.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Optimizer")

    # Required arguments
    parser.add_argument("--save_dir",      type=str, required=True,  help="Directory to save model checkpoints")
    parser.add_argument("--dataset_name",  type=str, required=True,  help="Dataset name: 'MNIST' or 'CIFAR10'")

    # Optional arguments with defaults
    parser.add_argument("--K",                  type=int,   default=64,     help="Output dimension of deep network")
    parser.add_argument("--num_para_min",       type=int,   default=10,     help="Minimum number of parameters to update per step")
    parser.add_argument("--num_para_max",       type=int,   default=200000,   help="Maximum number of parameters to update per step")
    parser.add_argument("--N",                  type=int,   default=3000,   help="Batch size")
    parser.add_argument("--sigma",              type=float, default=0.02,   help="Perturbation std dev")
    parser.add_argument("--beta",               type=float, default=0.5,    help="Beta for valid_dims term")
    parser.add_argument("--base_width",         type=int,   default=8,      help="Base width of ResNet9 (all datasets, including CIFAR10)")
    parser.add_argument("--num_epochs",         type=int,   default=5001,  help="Number of epochs to run")
    parser.add_argument("--learning_rate",      type=float, default=0.005,  help="ES learning rate / step size")
    parser.add_argument("--population_size",    type=int,   default=32,     help="ES population size (must be even)")
    parser.add_argument("--weight_decay",       type=float, default=0.005,  help="Weight decay")
    parser.add_argument("--consider_active_dim",type=lambda x: x.lower() != "false", default=True, help="Whether to consider active dimensions (default: True)")
    parser.add_argument("--kl_threshold",       type=float, default=0.005,  help="KL divergence threshold")
    parser.add_argument("--train_batch_size",   type=int,   default=128,    help="Training batch size")
    parser.add_argument("--num_train_epochs",   type=int,   default=4,      help="Number of training epochs for supervised fine-tuning")
    parser.add_argument("--train_start_epoch_1",  type=int,   default=2000,   help="Epoch to start first supervised fine-tuning")
    parser.add_argument("--train_start_epoch_2",  type=int,   default=4000,   help="Epoch to start second supervised fine-tuning")
    parser.add_argument("--in_between_es_epoch_1", type=int, default=500,   help="First interval for ES updates")
    parser.add_argument("--in_between_es_epoch_2", type=int, default=25,    help="Second interval for ES updates")
    parser.add_argument("--number_of_experiments", type=int,   default=5,     help="Number of experiments to run")
    parser.add_argument("--init_from", type=str, default=None,
                        help="Path to a checkpoint whose weights every experiment starts from "
                             "(e.g. a model trained by main_softmax.py). Unlike --resume the epoch "
                             "counter still starts at 0, so the full ES + fine-tuning schedule is "
                             "applied, and the collapsed-init check is skipped.")
    parser.add_argument("--resume", action="store_true", help="Resume each experiment from optimal_autoencoder.pth in its <save_dir>N/ (continues toward --num_epochs). Default OFF = fresh start, identical to previous behaviour.")
    args = parser.parse_args()


    for experiment in range(args.number_of_experiments):

        current_save_dir=args.save_dir + str(experiment) + '/'
        
        optimizer = Optimizer(
            K=args.K,
            num_para_min=args.num_para_min,
            num_para_max=args.num_para_max,
            N=args.N,
            sigma=args.sigma,
            beta=args.beta,
            base_width=args.base_width,
            dataset_name=args.dataset_name,
            save_dir=current_save_dir,
            learning_rate=args.learning_rate,
            population_size=args.population_size,
            weight_decay=args.weight_decay,
            consider_active_dim=args.consider_active_dim,
            num_epochs=args.num_epochs,
            kl_threshold=args.kl_threshold,
            train_batch_size=args.train_batch_size,
            num_train_epochs=args.num_train_epochs,
            train_start_epoch_1=args.train_start_epoch_1,
            train_start_epoch_2=args.train_start_epoch_2,
            in_between_es_epoch_1=args.in_between_es_epoch_1,
            in_between_es_epoch_2=args.in_between_es_epoch_2,
            resume=args.resume,
            init_from=args.init_from
        )
        optimizer.optimize()

        del optimizer
        gc.collect()
        torch.cuda.empty_cache()



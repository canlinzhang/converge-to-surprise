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
from scripts.pair_maker import PairMaker
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
'''
class Optimizer:
    def __init__(self, K=64, num_para_min=10, num_para_max=2000, N=3000, sigma=0.02, beta=0.5, base_width=8, dataset_name='MNIST',
                 learning_rate = 0.005, population_size = 30, weight_decay = 0.005,
                 save_dir='./model_bin/my_run/', consider_active_dim=True, num_epochs=10000, kl_threshold=0.005, train_batch_size=128, num_train_epochs=5, 
                 train_start_epoch_1=2000, train_start_epoch_2=4000, in_between_es_epoch_1=500, in_between_es_epoch_2=25):
        
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.dataset_name in ("MNIST", "FashionMNIST", "USPS"):
            self.in_channels = 1
        elif self.dataset_name == 'CIFAR10':
            self.in_channels = 3
        else:
            raise ValueError("Unsupported dataset. For now we only accept 'MNIST', 'FashionMNIST', 'USPS', and 'CIFAR10'.")

        # Instantiate once — shared weights for both views
        self.optimal_autoencoder = ResNet9(K=self.K, in_channels=self.in_channels, base_width=self.base_width, normalize=True).to(self.device)
        
        # Load latest checkpoint and meta_info if available
        self.start_epoch = 0

        #load the pair maker
        self.pair_maker = PairMaker(dataset_name=self.dataset_name, batch_size=self.N)

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

    def _check_and_reinit(self, max_tries=100):
        """Re-initialize until we get a non-collapsed starting point."""
        dummy_batch = next(iter(self.pair_maker.split_patches_from_loader(
            split="local_search", device=self.device)))
        tensor_i = dummy_batch[0]
        
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
            self.optimal_autoencoder = ResNet9(
                K=self.K, in_channels=self.in_channels,
                base_width=self.base_width, normalize=True
            ).to(self.device)
            self.es.autoencoder_original = self.optimal_autoencoder
            self.es.total_ids = sum(p.numel() for p in self.optimal_autoencoder.parameters())

    def optimize(self):
        # Move optimal autoencoder to GPU before starting optimization
        self.optimal_autoencoder = self.optimal_autoencoder.to(self.device)

        if self.start_epoch == 0:  # only check on fresh start, not resume
            self._check_and_reinit()

        # Adaptive num_para setup
        current_num_para = self.num_para_min

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):

            current_num_para = min(int(self.es.total_ids * ((epoch+1) / 100)), self.es.total_ids)

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
    parser.add_argument("--num_para_max",       type=int,   default=2000000,   help="Maximum number of parameters to update per step")
    parser.add_argument("--N",                  type=int,   default=3000,   help="Batch size")
    parser.add_argument("--sigma",              type=float, default=0.02,   help="Perturbation std dev")
    parser.add_argument("--beta",               type=float, default=0.5,    help="Beta for valid_dims term")
    parser.add_argument("--base_width",         type=int,   default=8,      help="Base width of ResNet9")
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
            in_between_es_epoch_2=args.in_between_es_epoch_2
        )
        optimizer.optimize()

        del optimizer
        gc.collect()
        torch.cuda.empty_cache()



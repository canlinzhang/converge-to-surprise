import os
import numpy as np
import math
import random
import argparse
import torch
import copy

from torch.func import functional_call

from copy import deepcopy
from functools import reduce
from collections import defaultdict, Counter
from scripts.surprise_score import SurpriseScore

#####################################
#class function to obtain index for parameters
class ObtainIndexForParameters():
    '''
    The idea is:
    1. Obtain the layer name and shape of each layer.
    2. For each layer, obtain the original index range.
    3. Build index list for shuffle.
    4. In optimization, given an index, use index range to locate its layer and position.
    '''
    def __init__(self, autoencoder):

        self.autoencoder = autoencoder

    #obtain the layer name for encoder and decoder
    #also get the ID range for each layer
    #original index range:
    #Layer 1: ID1 to ID2
    #Layer 2: ID2+1 to ID3, etc.
    def obtain_layer_name(self):
        Dict = defaultdict()
        ID_s = 0
        for name, param in self.autoencoder.named_parameters():
            # Modify the name to remove the double 'encoder' or 'decoder' prefix
            name_parts = name.split('.')
            if name_parts[0] == 'encoder':
                # Remove the first 'encoder' and retain the rest
                name = '.'.join(name_parts[1:])
            elif name_parts[0] == 'decoder':
                # Remove the first 'decoder' and retain the rest
                name = '.'.join(name_parts[1:])
          
            List = list(param.shape)
            range = math.prod(List)
            Dict[name] = dict()
            Dict[name]['layer_shape'] = List
            Dict[name]['ID_range'] = [ID_s, ID_s + range - 1]
            ID_s += range

        #return both number of IDs and dictionary
        return(Dict, ID_s)
  

######################################
#class function to update autoencoder parameters
'''
The old functions like 
locate_and_calculate_position, 
calculate_position_from_id, 
get_parameter_value, set_parameter_value, update_parameters_by_indices etc
are not used anymore in favor of the new flat parameter interface provided by get_flat and set_flat.
So we can use GPU for parameter udpates.
'''
class UpdateParameters():

    def __init__(self, layer_dict):

        self.layer_dict = layer_dict

    #given an index, we will locate its corresponding layer.
    def locate_layer_by_id(self, ID):
        for name in self.layer_dict:
            if self.layer_dict[name]['ID_range'][0] <= ID <= self.layer_dict[name]['ID_range'][1]:
                return name
        raise ValueError('ID out of range: do not match the parameter number of the AE')

    def calculate_position_from_id(self, layer_name, tensor_shape, relative_id):
        """
        Calculate the position in a tensor given a query ID.

        Args:
            layer_name (str): Name of the layer (not used in calculation but kept for consistency)
            tensor_shape (list): Shape of the tensor [N, C, H, W] for CNN, [out_features, in_features] for linear, or [D] for 1D
            relative_id (int): relative parameter position

        Returns:
            list: Position corresponding to the query_id
        """
        if len(tensor_shape) == 4:
            # CNN layer
            N, C, H, W = tensor_shape
            # Calculate position
            w = relative_id % W
            h = (relative_id // W) % H
            c = (relative_id // (W * H)) % C
            n = (relative_id // (W * H * C)) % N
            return [n, c, h, w]
        elif len(tensor_shape) == 2:
            # Linear/Dense layer
            out_features, in_features = tensor_shape
            # Calculate position
            in_idx = relative_id % in_features
            out_idx = (relative_id // in_features) % out_features
            return [out_idx, in_idx]
        elif len(tensor_shape) == 1:
            # One-dimensional tensor (bias or batch norm)
            D = tensor_shape[0]
            d = relative_id % D
            return [d]
        else:
            raise ValueError("Unsupported tensor shape: {}".format(tensor_shape))

    def locate_and_calculate_position(self, query_id):
        """
        Locate the layer and calculate the position in the tensor for a given ID.
      
        Args:
            layer_dict (dict): Dictionary containing layer information.
            query_id (int): The ID to locate and convert to position.
      
        Returns:
            tuple: (layer_name, position) corresponding to the query_id
        """
        # Locate the layer name using the query ID
        layer_name = self.locate_layer_by_id(query_id)
      
        # Get the ID range for the located layer
        start_id = self.layer_dict[layer_name]['ID_range'][0]
        # Calculate the actual ID within the layer's range
        relative_id = query_id - start_id
      
        # Calculate the position in the tensor
        tensor_shape = self.layer_dict[layer_name]['layer_shape']
        position = self.calculate_position_from_id(layer_name, tensor_shape, relative_id)
      
        return layer_name, position

    def get_parameter_value(self, autoencoder, layer_name, position):
        """
        Get the value of the parameter given the layer name and position.

        Args:
            layer_name (str): The name of the layer.
            position (list): The position of the parameter in the tensor.

        Returns:
            float: The value of the parameter at the specified position.
        """
        # Find the parameter tensor in the autoencoder
        for name, param in autoencoder.named_parameters():
            if name.endswith(layer_name):
                # Convert position list to a tuple for indexing
                pos_tuple = tuple(position)
                return param.data[pos_tuple].item()
        raise ValueError("Layer not found in the autoencoder: {}".format(layer_name))

    def set_parameter_value(self, autoencoder, layer_name, position, new_value):
        """
        Set the value of the parameter given the layer name and position.

        Args:
            layer_name (str): The name of the layer.
            position (list): The position of the parameter in the tensor.
            new_value (float): The new value to set.

        """
        # Find the parameter tensor in the autoencoder
        for name, param in autoencoder.named_parameters():
            if name.endswith(layer_name):
                # Convert position list to a tuple for indexing
                pos_tuple = tuple(position)
                param.data[pos_tuple] = new_value
                return
        raise ValueError("Layer not found in the autoencoder: {}".format(layer_name))
    
    def keep_original_parameter_value(self, autoencoder, indices):
        """
        Keep a copy of the original parameter values for the given indices.

        Args:
            autoencoder (nn.Module): The autoencoder model.
            indices (list): List of parameter indices to keep original values for.

        Returns:
            dict: Mapping from index to original parameter value.
        """
        original_values = {}
        for query_k in indices:
            # Locate layer and position
            layer_name, position = self.locate_and_calculate_position(query_k)

            # Get current value
            value_k = self.get_parameter_value(autoencoder, layer_name, position)
            
            original_values[query_k] = value_k

        return original_values

    def revert_parameters(self, autoencoder, original_values):
        """
        Revert the parameters of the autoencoder to the original values
        stored in original_values.

        Args:
            autoencoder (nn.Module): The autoencoder model.
            original_values (dict): Mapping from parameter index to original value.
        """
        for query_k, value_k in original_values.items():
            # Locate layer and position
            layer_name, position = self.locate_and_calculate_position(query_k)

            # Set the parameter back to its original value
            self.set_parameter_value(autoencoder, layer_name, position, value_k)

        return autoencoder

    def update_parameters_by_indices(self, autoencoder, indices, perturbations, sigma, multiply_factor):
        """
        Update the parameters corresponding to a list of indices.
      
        Args:
            indices (list): List of parameter indices to update.
            sigma (float): Standard deviation for Gaussian perturbations.
            multiply_factor (float): Factor to multiply the current parameter value by before adding the perturbation.
      
        Returns:
            autoencoder: The updated autoencoder model.
        """

        for i, query_k in enumerate(indices):
            # Locate layer and position
            layer_name, position = self.locate_and_calculate_position(query_k)

            # Get current value
            value_k = self.get_parameter_value(autoencoder, layer_name, position)

            # Perturb: theta + sigma * epsilon_i
            self.set_parameter_value(
                autoencoder, layer_name, position,
                (multiply_factor * value_k) + (sigma * perturbations[i])
            )

        return autoencoder

    def keep_original_parameter_value_GPU(self, autoencoder, indices):
        """
        Keep a copy of the original parameter values for the given indices.

        Args:
            autoencoder (nn.Module): The autoencoder model.
            indices (np.ndarray): Selected parameter indices, shape (num_para,).

        Returns:
            tuple: (theta, idx_tensor) — full flat GPU tensor and index tensor.
        """
        device = next(autoencoder.parameters()).device

        theta = torch.cat([p.data.view(-1) for p in autoencoder.parameters()])
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)

        # Save only the selected values (not the full theta)
        original_values = theta[idx_tensor].clone()

        return theta, idx_tensor, original_values


    def revert_parameters_GPU(self, autoencoder, theta_input, idx_tensor, original_values):
        """
        Revert the parameters at the selected indices back to original values.

        Args:
            autoencoder (nn.Module): The autoencoder model.
            theta (torch.Tensor): Full flat parameter vector on GPU.
            idx_tensor (torch.Tensor): Selected indices as a GPU tensor.
            original_values (torch.Tensor): Original values at those indices.
        """
        theta = theta_input.clone()
        theta[idx_tensor] = original_values

        # Write back
        offset = 0
        for p in autoencoder.parameters():
            size = p.data.numel()
            p.data.copy_(theta[offset:offset + size].view(p.data.shape))
            offset += size

        return autoencoder

    def update_parameters_by_indices_GPU(self, autoencoder, theta_input, idx_tensor, perturbations, sigma, multiply_factor):
        """
        GPU-accelerated parameter update for selected indices.
        
        Args:
            autoencoder: The model to update.
            idx_tensor (torch.Tensor): Selected parameter indices as a GPU tensor, shape (num_para,).
            perturbations (torch.Tensor): Noise vector as a GPU tensor, shape (num_para,).
            sigma (float): Noise standard deviation.
            multiply_factor (float): Weight decay factor applied to current values.
        
        Returns:
            autoencoder: The updated model.
        """

        # Get full flat parameter vector on GPU
        theta = theta_input.clone()  # <-- add .clone()

        # (multiply_factor * theta[indices]) + (sigma * eps)
        theta[idx_tensor] = multiply_factor * theta[idx_tensor] + sigma * perturbations

        # Write back
        offset = 0
        for p in autoencoder.parameters():
            size = p.data.numel()
            p.data.copy_(theta[offset:offset + size].view(p.data.shape))
            offset += size

        return autoencoder


##################################
### evaluation code ##############
def eval_score(autoencoder, tensor_i, tensor_j, K, beta, consider_active_dim):
    """Forward pass + surprise score for one autoencoder."""
    with torch.no_grad():
        emb_i = autoencoder(tensor_i)
        emb_j = autoencoder(tensor_j)
        seq_i = emb_i.argmax(dim=-1).cpu().tolist()
        seq_j = emb_j.argmax(dim=-1).cpu().tolist()

    scorer = SurpriseScore(seq_i, seq_j, K)
    score  = scorer.score("kl")
    score_per_dim  = scorer.score("per_dim_kl")
    valid  = scorer.score("valid_dims")
    agreement = scorer.score("view_agreement")
    
    #final  = score * np.log1p(beta * valid) if consider_active_dim else score
    #final = score_per_dim
    return score_per_dim, score, valid, agreement


def split_seq_by_values(seq_0, seq_1, ID):
    """
    Split seq_1 into sub-sequences based on value groupings in seq_0.

    Args:
        seq_0: list of non-negative integers (used as grouping keys)
        seq_1: list of values to be split

    Returns:
        dict mapping each unique value in seq_0 to its corresponding
        sub-sequence extracted from seq_1 (order preserved)
    """
    # Step 1: Build position map {value: [positions]} from seq_0
    position_map = {}
    for pos, val in enumerate(seq_0):
        if val not in position_map:
            position_map[val] = []
        position_map[val].append(pos)

    # Step 2: For each value, extract seq_1 elements at those positions
    result = {}
    for val, positions in position_map.items():
        result[(ID,val)] = [seq_1[p] for p in positions]

    return result


def eval_score_new(autoencoder, old_autoencoders, tensor_i, tensor_j, K, beta, consider_active_dim):
    """Forward pass + surprise score for one autoencoder."""
    with torch.no_grad():
        emb_i = autoencoder(tensor_i)
        emb_j = autoencoder(tensor_j)
        seq_i = emb_i.argmax(dim=-1).cpu().tolist()
        seq_j = emb_j.argmax(dim=-1).cpu().tolist()

        #we keep old autoencocers in a list
        #if the list is not empty, we proceed.
        if len(old_autoencoders) > 0: 
            result_i, result_j = dict(), dict()
            for ID, old_autoencoder in enumerate(old_autoencoders):
                old_emb = old_autoencoder(tensor_i)
                old_seq = old_emb.argmax(dim=-1).cpu().tolist()

                result_i_temp = split_seq_by_values(old_seq, seq_i, ID)
                result_j_temp = split_seq_by_values(old_seq, seq_j, ID)
                result_i.update(result_i_temp)
                result_j.update(result_j_temp)

    scorer = SurpriseScore(seq_i, seq_j, K)
    score  = scorer.score("kl")
    score_per_dim  = scorer.score("per_dim_kl")
    valid  = scorer.score("valid_dims")
    agreement = scorer.score("view_agreement")

    if len(old_autoencoders) > 0:
        for key in result_i:
            if key not in result_j or len(result_i[key]) != len(result_j[key]):
                raise ValueError(f"Key {key} missing or length mismatch in result_j")
            sub_scorer = SurpriseScore(result_i[key], result_j[key], K)
            
            sub_score = sub_scorer.score("kl")
            score += sub_score * (len(result_i[key]) / len(seq_i))  # weight by sub-sequence length
            
            sub_score_per_dim = sub_scorer.score("per_dim_kl")
            score_per_dim += sub_score_per_dim * (len(result_i[key]) / len(seq_i))  # weight by sub-sequence length



def compute_centered_ranks(x):
    """
    Convert raw fitness values into centered ranks in [-0.5, 0.5].
    This is the standard ES rank shaping trick.
    """
    x = np.asarray(x)
    ranks = np.empty(len(x), dtype=np.float32)
    ranks[np.argsort(x)] = np.arange(len(x), dtype=np.float32)
    ranks /= (len(x) - 1)
    ranks -= 0.5
    return ranks


##################################
### the main evolution strategy ##
class EvolutionStrategy:
    def __init__(self, autoencoder_original, sigma, learning_rate, population_size,
                        K, beta, weight_decay, consider_active_dim, device, chunk_size=16):
        """
        Initialize the evolution strategy.

        Args:
            autoencoder      : the base autoencoder to perturb
            sigma            : standard deviation for Gaussian perturbations
            learning_rate    : step size for parameter updates (alpha in original OpenAI ES paper)
            population_size  : number of perturbations per generation (P, even).
                               The population is evaluated with torch.vmap, in
                               chunks of `chunk_size` populations per launch.
            K                : number of output dimensions
            beta             : weight for valid_dimensions term
            weight_decay     : L2 regularization factor applied to parameter updates (lambda in original OpenAI ES paper)
            consider_active_dim : whether to multiply surprise score by log1p(beta * valid_dimension)
            device           : torch device to run the autoencoder on
            chunk_size       : number of perturbed populations evaluated in a single
                               torch.vmap launch. The population is split into
                               ceil(P / chunk_size) chunks that are run sequentially
                               and concatenated, which lowers peak GPU memory (and
                               keeps the vmap BatchNorm tensor under cuDNN's 2^31
                               element limit) WITHOUT changing the result: every
                               population is scored independently, so chunking only
                               changes how many run at once. Default 16 → P=32 runs
                               as two chunks of 16. Set >= population_size for a
                               single full-population launch.
        """

        self.sigma = sigma
        self.learning_rate = learning_rate
        self.population_size = population_size
        self.K = K
        self.beta = beta #beta is for the valid_dimensions term in the final surprise score computation
        self.weight_decay = weight_decay
        self.consider_active_dim = consider_active_dim
        self.device = device
        self.chunk_size = chunk_size

        # Obtain layer_dict and number of total parameters
        self.helper = ObtainIndexForParameters(autoencoder_original)
        self.layer_dict, self.total_ids = self.helper.obtain_layer_name()
      
        # Initialize updater
        self.updater = UpdateParameters(self.layer_dict)

    def _forward_populations(self, autoencoder, theta_stacked, x):
        """
        Evaluate P0 perturbed copies of `autoencoder` on the SAME input batch in
        parallel via torch.vmap + functional_call — one fused GPU launch instead
        of a Python loop over the population.

        Args:
            autoencoder   : the base module (supplies architecture + current
                            train/eval mode). functional_call swaps its params/
                            buffers out per population and leaves the module's own
                            tensors unchanged on return.
            theta_stacked : (P0, total_ids) flat parameter vectors, one row per
                            population (already perturbed).
            x             : (B, C, H, W) input batch, shared by all P0 populations.

        Returns:
            (P0, B, K) stacked logits.

        BatchNorm note: during ES the base model is in TRAIN mode, so each
        population computes its OWN per-batch BN statistics (identical to the old
        sequential forward). functional_call swaps in per-population stacked
        buffers; BN's in-place running-stat update lands on those throwaway copies
        and is discarded — the module's real running buffers are refreshed
        separately at the end of run().
        """
        P0 = theta_stacked.shape[0]

        # Flat (P0, total_ids) → per-parameter stacked dict {name: (P0, *shape)}.
        # named_parameters() iterates in the same order used to build the flat
        # theta, so the offsets line up.
        params = {}
        offset = 0
        for name, p in autoencoder.named_parameters():
            n = p.numel()
            params[name] = theta_stacked[:, offset:offset + n].reshape(P0, *p.shape)
            offset += n

        # Per-population buffer copies {name: (P0, *shape)} (BN running stats,
        # num_batches_tracked, ...). Same values for every population.
        buffers = {}
        for name, b in autoencoder.named_buffers():
            rep = [P0] + [1] * b.dim()
            buffers[name] = b.detach().unsqueeze(0).repeat(*rep).contiguous()

        # Same input batch for every population (no copy — expand is a view).
        x_rep = x.unsqueeze(0).expand(P0, *x.shape)

        def fmodel(p_dict, b_dict, inp):
            return functional_call(autoencoder, (p_dict, b_dict), (inp,))

        return torch.vmap(fmodel)(params, buffers, x_rep)   # (P0, B, K)

    def _forward_populations_chunked(self, autoencoder, theta_stacked, x):
        """
        Same as `_forward_populations`, but splits the P populations into chunks of
        `self.chunk_size` and evaluates them in separate torch.vmap launches, then
        concatenates the per-chunk logits back into one (P, B, K) tensor.

        Populations are independent under vmap (each carries its own perturbed
        params and its own per-batch BatchNorm statistics), so the row order and
        the per-population outputs are identical to a single full-population launch;
        chunking only reduces how many run concurrently, lowering peak memory and
        keeping the vmap BN tensor under cuDNN's 2^31-element limit.
        """
        P0 = theta_stacked.shape[0]
        chunk = self.chunk_size if self.chunk_size and self.chunk_size > 0 else P0

        if chunk >= P0:
            return self._forward_populations(autoencoder, theta_stacked, x)

        outs = []
        for start in range(0, P0, chunk):
            end = min(start + chunk, P0)
            outs.append(self._forward_populations(autoencoder, theta_stacked[start:end], x))
        return torch.cat(outs, dim=0)   # (P0, B, K)

    def run(self, autoencoder, tensor_i, tensor_j, num_para):
        """
        One ES generation, the population evaluated in parallel with torch.vmap
        in chunks of self.chunk_size perturbed networks per launch (results are
        identical to a single full-population launch; chunking only lowers peak
        memory). All populations share:
          * the SAME randomly-selected parameter indices to perturb — re-sampled
            fresh each call/epoch, fixed within the call, and
          * the SAME input batch (tensor_i / tensor_j).
        Mirror sampling: we draw P/2 epsilons and append their negatives, so the
        population holds matched ± pairs. Centered-rank shaping and the final
        weighted-direction update match the original sequential implementation.
        """
        P = self.population_size
        if P % 2 != 0:
            raise ValueError("population_size must be even for mirror sampling")

        device = self.device

        # Randomly select which parameters to perturb — fixed for this call,
        # shared by every population.
        indices = np.array(random.sample(range(self.total_ids), num_para))

        # Flat parameter vector (theta) + selected-index tensor on GPU.
        theta, idx_tensor, original_values = self.updater.keep_original_parameter_value_GPU(autoencoder, indices)

        with torch.no_grad():
            # P/2 epsilons + their mirrors → (P, num_para).
            eps     = torch.randn(P // 2, num_para, device=device, dtype=theta.dtype)
            perturb = torch.cat([eps, -eps], dim=0)

            # P perturbed flat parameter vectors: theta + sigma * eps applied to
            # the selected indices only.
            theta_stacked = theta.unsqueeze(0).repeat(P, 1)                  # (P, total_ids)
            theta_stacked[:, idx_tensor] = theta_stacked[:, idx_tensor] + self.sigma * perturb

            # Parallel forward on both complementary views, in chunks of
            # self.chunk_size populations per vmap launch (results identical to a
            # single full-population launch; see _forward_populations_chunked).
            emb_i = self._forward_populations_chunked(autoencoder, theta_stacked, tensor_i)   # (P, B, K)
            emb_j = self._forward_populations_chunked(autoencoder, theta_stacked, tensor_j)

            seq_i = emb_i.argmax(dim=-1).cpu().numpy()   # (P, B)
            seq_j = emb_j.argmax(dim=-1).cpu().numpy()

        # Surprise score per population — same calculation as eval_score.
        all_scores = []
        for p in range(P):
            scorer = SurpriseScore(seq_i[p], seq_j[p], self.K)
            all_scores.append(scorer.score("per_dim_kl"))
            del scorer

        # Centered-rank shaping over the population, then weighted update.
        ranked_scores = compute_centered_ranks(all_scores)
        ranked_scores = torch.tensor(ranked_scores, dtype=torch.float32, device=device)

        weighted_eps = torch.sum(ranked_scores[:, None] * perturb, axis=0)   # (num_para,)

        delta = (self.learning_rate / (P * self.sigma)) * weighted_eps

        # apply weight decay
        decay_factor = 1.0 - self.weight_decay * self.learning_rate

        autoencoder = self.updater.update_parameters_by_indices_GPU(
            autoencoder, theta, idx_tensor, delta, sigma=1.0, multiply_factor=decay_factor
        )

        # Refresh BatchNorm running buffers. The parallel scoring forwards above
        # discard their buffer updates (functional_call is stateless), so do ONE
        # train-mode forward with the UPDATED weights to keep running stats
        # populated for the eval-mode passes used by FT-Phase-1 and test.py. (The
        # old sequential code updated buffers as a side effect of its per-
        # population forwards; one clean forward per call serves the same role.)
        with torch.no_grad():
            autoencoder(tensor_i)

        return autoencoder





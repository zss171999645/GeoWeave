import numpy as np
from logging import warning
from math import ceil
import random
from typing import Callable
import warnings
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.math_utils import affine_padding, affine_inverse
from easyvolcap.dataloaders.datasets.src_view_selector.base import BaseSelector
import torch


class BaseMultiCamSelector(BaseSelector):
    def __init__(self, src_exts: torch.Tensor, random_ap: int, is_train: bool):
        super().__init__(src_exts, random_ap, is_train)
    
    def sample_n(self, ind_list, extra_list, n_srcs):
        assert len(ind_list) == len(extra_list), "The number of indices and extra indices should be the same"
        if self.random_ap:
            if n_srcs < len(ind_list): inds = random.sample(list(range(len(ind_list))), n_srcs)  # (S,), no replacement
            else: inds = random.choices(list(range(len(ind_list))), k=n_srcs)  # (S,), with replacement
            ind_list = [ind_list[i] for i in inds]
            extra_list = [extra_list[i] for i in inds]
        else:
            assert len(ind_list) == n_srcs, "The number of source views should be equal to n_srcs"
        return ind_list, extra_list


class MultiviewSeqSrcviewSelector(BaseMultiCamSelector):
    def __init__(self, src_exts: torch.Tensor, random_ap: int, is_train: bool, source_cfg: dotdict = None):
        super().__init__(src_exts, random_ap, is_train)
        if not self.is_train:
            if self.random_ap != 0:
                warnings.warn(f"MultiviewSeqSrcviewSelector: random_ap should be 0 in eval mode, got {self.random_ap}")
                self.random_ap = 0

    def __call__(self, target_index, extra_index, n_srcs, output: dotdict = None):
        # TODO: find a better way to do this, it is not elegant...
        # Select the closest view along the temporal dimension centered at the target view
        n_srcs_sample = int(ceil((n_srcs + 1) / self.n_cams))  # use random_ap to sample temporal inds
        sidx = max(0, target_index - (n_srcs_sample + self.random_ap) // 2)
        inds = list(range(sidx, min(sidx + n_srcs_sample + self.random_ap, self.n_views)))  # (random_ap + 1,)

        inds.remove(target_index)  # (S + random_ap,), remove itself                
        src_inds = inds.copy()

        extra_index, inds = self.compute_mvseq_inds(self.n_cams, src_inds, target_index, n_srcs, self.is_train)
        
        return inds, extra_index

    def compute_mvseq_inds(self, n_cam, src_inds, target_index, n_srcs, is_train=True):
        # keep the front view and target frame
        src_inds = [target_index] + src_inds   # add target index
        all_extra_index = [[x] * len(src_inds) for x in range(n_cam)]
        all_extra_index = [item for sublist in all_extra_index for item in sublist]
        all_inds = src_inds * n_cam

        if len(all_inds) - 1 < n_srcs:
            warnings.warn(f"MultiviewSeqSrcviewSelector: n_srcs > len(all_inds), got {n_srcs} > {len(all_inds)}")

        if is_train:
            all_extra_index = all_extra_index[1:]  # remove target camera index
            all_inds = all_inds[1:]  # remove target frame index
            if n_srcs <= len(all_inds):
                sample_idx = random.sample(range(len(all_inds)), n_srcs)  # near and far frames are randomly sampled
            else: 
                warnings.warn(f"MultiviewSeqSrcviewSelector: n_srcs > len(all_inds), got {n_srcs} > {len(all_inds)}")
                sample_idx = random.choices(range(len(all_inds)), k=n_srcs - len(all_inds)) + list(range(len(all_inds)))
            extra_index = [all_extra_index[i] for i in sample_idx]
            inds = [all_inds[i] for i in sample_idx]
        else:
            all_extra_index = all_extra_index[1:]  # remove target camera index
            all_inds = all_inds[1:]  # remove target frame index
            if n_srcs <= len(all_inds):
                extra_index = all_extra_index[:n_srcs]
                inds = all_inds[:n_srcs]
            else:
                warnings.warn(f"MultiviewSeqSrcviewSelector: n_srcs > len(all_inds), got {n_srcs} > {len(all_inds)}")
                sample_idx = random.choices(range(len(all_inds)), k=n_srcs - len(all_inds)) + list(range(len(all_inds)))
                extra_index = [all_extra_index[i] for i in sample_idx]
                inds = [all_inds[i] for i in sample_idx]
        return extra_index, inds


class PairwiseGraphSrcviewSelector(BaseSelector):
    def __init__(self, src_exts: torch.Tensor, random_ap: int, is_train: bool, source_cfg: dotdict = None):
        super().__init__(src_exts, random_ap, is_train)
        self.covis_matrix = source_cfg["covis_matrix"]
        self.covis_dist_matrix = source_cfg["covis_dist_matrix"]
        self.select_top_prob = source_cfg.get("select_top_prob", 1.0)
        self.top_thr = source_cfg.get("top_thr", 0.2)
        self.select_loop_prob = source_cfg.get("select_loop_prob", 1.0)
        self.loop_thr = source_cfg.get("loop_thr", 0.3)
        self.loop_frames_min_gap = source_cfg.get("loop_frames_min_gap", 30)
        self.loop_frames_max_dist = source_cfg.get("loop_frames_max_dist", 15.0)

    def __call__(self, target_index, extra_index, n_srcs, output: dotdict = None):
        extra_index, inds, _, covis_matrix, covis_indices, covis_pairs = \
            self.compute_pairs_inds(self.covis_matrix, self.covis_dist_matrix, self.n_views, target_index, extra_index, n_srcs, 0, 4,
                                    select_top_prob=self.select_top_prob, top_thr=self.top_thr,
                                    select_loop_prob=self.select_loop_prob, loop_thr=self.loop_thr,
                                    loop_frames_min_gap=self.loop_frames_min_gap, loop_frames_max_dist=self.loop_frames_max_dist,
                                    is_train=self.is_train)
        output.covis_matrix = covis_matrix
        output.covis_indices = covis_indices
        output.covis_pairs = covis_pairs
        output.covis_dist = torch.abs(inds[output.covis_pairs[:, 0]] - inds[output.covis_pairs[:, 1]] - 0.01).float().unsqueeze(1).repeat(1, 2)
        return inds, extra_index

    def convert_samples_to_inds(self, samples, n_views):
        extra_index = []  # view idx
        inds = []  # frame idx
        for sample in samples:
            extra_index.append(sample // n_views)
            inds.append(sample % n_views)
        extra_index = torch.as_tensor(extra_index)
        inds = torch.as_tensor(inds)
        return extra_index, inds

    def compute_pairs_inds(self, covis_matrix, covis_dist_matrix, n_views, target_index, extra_index, n_srcs, covis_thres=0, max_retries=4, 
                           select_top_prob=1.0, top_thr=0.2, select_loop_prob=1.0, loop_thr=0.3, 
                           loop_frames_min_gap=30, loop_frames_max_dist=15.0, is_train=True):
        cam_index = extra_index
        frame_index = target_index
        start_index = int(cam_index * n_views + frame_index)
        num_samples = int(n_srcs + 1)
        covis_matrix = covis_matrix.clone()
        covis_dist_matrix = covis_dist_matrix.clone()
        valid_train_sample = False
        if is_train:
            covis_matrix = covis_matrix.cpu().numpy()
            covis_dist_matrix = covis_dist_matrix.cpu().numpy()
            pairwise_dist_start = covis_dist_matrix[start_index, :]
            num_nodes = covis_matrix.shape[0]
            excluded_nodes = set()
            best_walk = []  # To keep track of the best walk found
            for _ in range(max_retries):
                visited = set()
                walk = []  # List to store the random walk sampling order
                stack = []  # Stack for backtracking

                # Choose a random starting index that is not in the excluded set
                all_nodes = set(range(num_nodes))
                available_nodes = list(all_nodes - excluded_nodes)
                if not available_nodes:
                    break  # No more nodes to try
                start = start_index
                walk.append(start)
                visited.add(start)
                stack.append(start)

                # Continue until we have sampled S indices or all expandable nodes are exhausted
                while len(walk) < num_samples and stack:
                    current = stack[-1]
                    # Get the pairwise covisibility and distance for the current node
                    pairwise_covisibility = covis_matrix[current, :]
                    
                    # Assign overlap score of zero to self-pairs
                    pairwise_covisibility[current] = 0
                    # Threshold the covisibility to get adjacency list for the current node
                    adjacency_list_for_current = (
                        pairwise_covisibility > covis_thres
                    ).astype(int)
                    adjacency_list_for_current = np.flatnonzero(adjacency_list_for_current)
                    # Get all unvisited neighbors
                    candidates = [
                        idx for idx in adjacency_list_for_current if idx not in visited
                    ]  # Remove visited nodes
                    candidates_covisibility = pairwise_covisibility[candidates]
                    if candidates:
                        if random.random() <= (1 - select_top_prob):
                            # Randomly select one of the unvisited overlapping neighbors
                            next_node = int(random.choice(candidates))
                        else:
                            # Select select one of the unvisited overlapping neighbors in top 20%
                            candidates = np.array(candidates)
                            find_loop = False
                            if random.random() <= select_loop_prob:
                                # Select a loop pair
                                sort_indices = np.argsort(candidates_covisibility)[::-1]
                                candidates_covisibility = candidates_covisibility[sort_indices]
                                candidates = candidates[sort_indices]
                                frame_inds = candidates % n_views
                                start_frame_ind = start_index % n_views
                                loop_mask = (np.abs(frame_inds - start_frame_ind) >= loop_frames_min_gap) & \
                                                (pairwise_dist_start[candidates] > 0) & \
                                                (pairwise_dist_start[candidates] <= loop_frames_max_dist)
                                top_len = int(len(candidates) * top_thr)
                                if np.sum(loop_mask) > 0:
                                    if top_len > 0 and (loop_mask[:top_len].sum() / top_len) >= loop_thr:
                                        candidates = candidates[loop_mask]
                                        next_node = int(random.choice(candidates))
                                        find_loop = True
                                else:
                                    find_loop = False
                            if not find_loop:
                                sort_indices = np.argsort(candidates_covisibility)[::-1]
                                candidates_covisibility = candidates_covisibility[sort_indices]
                                candidates = candidates[sort_indices].tolist()
                                top_len = int(len(candidates) * top_thr)
                                if top_len == 0:
                                    next_node = int(random.choice(candidates))
                                else:
                                    next_node = int(random.choice(candidates[:top_len]))

                        walk.append(next_node)
                        visited.add(next_node)
                        stack.append(next_node)
                    else:
                        # If no unvisited neighbor is available, backtrack
                        stack.pop()

                # Update the best walk if the current walk is larger
                if len(walk) > len(best_walk):
                    best_walk = walk

                # If we have enough samples, return the result
                if len(walk) >= num_samples:
                    break

                # Add all visited nodes to the excluded set
                excluded_nodes.update(visited)

            if len(best_walk) >= num_samples:
                valid_train_sample = True
                sample_indices = torch.tensor(best_walk[:num_samples])
                extra_index, inds = self.convert_samples_to_inds(sample_indices, n_views)
                sample_covis_matrix = torch.zeros((num_samples, num_samples), dtype=torch.long)
                for i in range(num_samples):
                    for j in range(num_samples):
                        if covis_matrix[sample_indices[i], sample_indices[j]] > 0:
                            sample_covis_matrix[i, j] = 1
                            sample_covis_matrix[j, i] = 1
                sample_covis_pairs = sample_covis_matrix.nonzero(as_tuple=False).contiguous().view(-1, 2)
                sample_covis_indices = torch.ones(sample_covis_pairs.shape[0], dtype=torch.long)
                sample_covis_pairs = sample_covis_pairs[:, [1, 0]]
                return extra_index, inds, sample_indices, sample_covis_matrix, sample_covis_indices, sample_covis_pairs

        if (not valid_train_sample) or (not is_train):
            if not isinstance(covis_matrix, torch.Tensor):
                covis_matrix = torch.from_numpy(covis_matrix)
            # multi view seqence
            n_cam = covis_matrix.shape[0] // n_views
            start_ind = max(0, frame_index - num_samples // 2 // n_cam)
            end_ind = min(start_ind + num_samples // n_cam + 1, n_views - 1)
            frame_inds = torch.arange(start_ind, end_ind)
            frame_inds = frame_inds[frame_inds != frame_index]
            frame_inds = torch.cat([torch.tensor([frame_index]), frame_inds])  # move to front
            cam_inds = torch.arange(n_cam)
            cam_inds = cam_inds[cam_inds != cam_index]
            cam_inds = torch.cat([torch.tensor([cam_index]), cam_inds])  # move to front
            sample_indices = []
            for cam_ind in cam_inds:
                for frame_index in frame_inds:
                    sample_indices.append(cam_ind * n_views + frame_index)
            sample_indices = torch.tensor(sample_indices)
            if len(sample_indices) >= num_samples:
                sample_indices = sample_indices[:num_samples]
            else:
                while len(sample_indices) < num_samples:
                    sample_indices_cp = sample_indices.clone()
                    sample_indices = torch.cat([sample_indices, sample_indices_cp])
                sample_indices = sample_indices[:num_samples]
            extra_index, inds = self.convert_samples_to_inds(sample_indices, n_views)
            sample_covis_matrix = torch.zeros((num_samples, num_samples), dtype=torch.long)
            for i in range(num_samples):
                for j in range(num_samples):
                    if covis_matrix[sample_indices[i], sample_indices[j]] > 0:
                        sample_covis_matrix[i, j] = 1
                        sample_covis_matrix[j, i] = 1
            if sample_covis_matrix.sum() == 0:
                sample_covis_matrix = (torch.abs(inds[:, None] - inds[None, :]) <= 5).long()
            sample_covis_pairs = sample_covis_matrix.nonzero(as_tuple=False).contiguous().view(-1, 2)
            sample_covis_indices = torch.ones(sample_covis_pairs.shape[0], dtype=torch.long)
            sample_covis_pairs = sample_covis_pairs[:, [1, 0]]
            
            return extra_index, inds, sample_indices, sample_covis_matrix, sample_covis_indices, sample_covis_pairs

import random
import torch


def compute_mvseq_inds(n_cam, src_inds, target_index, n_srcs, is_train=True, ref_view_type="front_ref_view"):
    # keep the front view and target frame
    src_inds = [target_index] + src_inds   # add target index
    all_extra_index = [[x] * len(src_inds) for x in range(n_cam)]
    all_extra_index = [item for sublist in all_extra_index for item in sublist]
    all_inds = src_inds * n_cam
    if is_train:
        if ref_view_type == "front_ref_view":
            ref_cam = 0
        elif ref_view_type == "random_front_ref_view":
            ref_cam = random.choice([0,6]) if n_cam > 6 else 0
        elif ref_view_type == "random_ref_view":
            ref_cam = random.choice(range(n_cam))
        else:
            raise NotImplementedError

        all_extra_index = all_extra_index[1:]  # remove target camera index
        all_inds = all_inds[1:]  # remove target frame index
        if n_srcs < len(all_inds): sample_idx = random.sample(range(len(all_inds)), n_srcs)  # near and far frames are randomly sampled
        else: 
            print("n_srcs < len(all_inds)", n_srcs, len(all_inds))
            sample_idx = random.choices(range(len(all_inds)), k=n_srcs)
        extra_index = [all_extra_index[i] for i in sample_idx]
        inds = [all_inds[i] for i in sample_idx]
        extra_index = [ref_cam] + extra_index  # add target camera index
        inds = [target_index] + inds  # add target frame index
    else:
        extra_index = all_extra_index
        inds = all_inds
    extra_index = torch.as_tensor(extra_index)
    inds = torch.as_tensor(inds)
    return extra_index, inds

def compute_mvseq_inds_with_loopclose(n_cam, src_inds, target_index, n_srcs, src_exts_3ddr_inv, is_train=True, ref_view_type="front_ref_view", max_distance = 3.0, min_id_gap = 50):
    c2ws_cam_front = src_exts_3ddr_inv[:,0,:,:]
    pos_cam_front = c2ws_cam_front[:, :3, 3]
    target_pos = pos_cam_front[target_index]
    distances = torch.norm(pos_cam_front - target_pos, dim=1)
    # 计算ID差距 [n_src_frames,]
    n_total_frames = len(distances)
    all_frame_indices = list(range(n_total_frames))
    
    # 转换为张量进行批量计算
    all_indices_tensor = torch.tensor(all_frame_indices, dtype=torch.long)
    id_gaps = torch.abs(all_indices_tensor - target_index)
    # 创建过滤掩码：距离小于阈值且ID差距大于阈值，且不是目标帧本身
    mask = (distances < max_distance) & (id_gaps > min_id_gap) & (all_indices_tensor != target_index)
    filtered_inds_tensor = all_indices_tensor[mask]
    filtered_distances = distances[mask]
    if len(filtered_inds_tensor) > 0:
        sorted_indices = torch.argsort(filtered_distances)
        sorted_filtered_inds = filtered_inds_tensor[sorted_indices].tolist()
    else:
        sorted_filtered_inds = []
    if len(sorted_filtered_inds) > 10:
        sorted_filtered_inds = sorted_filtered_inds[:10]
        

    src_inds = src_inds + sorted_filtered_inds
    # keep the front view and target frame
    src_inds = [target_index] + src_inds   # add target index
    all_extra_index = [[x] * len(src_inds) for x in range(n_cam)]
    all_extra_index = [item for sublist in all_extra_index for item in sublist]
    all_inds = src_inds * n_cam
    if is_train:
        if ref_view_type == "front_ref_view":
            ref_cam = 0
        elif ref_view_type == "random_front_ref_view":
            ref_cam = random.choice([0,6]) if n_cam > 6 else 0
        elif ref_view_type == "random_ref_view":
            ref_cam = random.choice(range(n_cam))
        else:
            raise NotImplementedError

        all_extra_index = all_extra_index[1:]  # remove target camera index
        all_inds = all_inds[1:]  # remove target frame index
        if n_srcs < len(all_inds): sample_idx = random.sample(range(len(all_inds)), n_srcs)  # near and far frames are randomly sampled
        else: 
            print("n_srcs < len(all_inds)", n_srcs, len(all_inds))
            sample_idx = random.choices(range(len(all_inds)), k=n_srcs)
        extra_index = [all_extra_index[i] for i in sample_idx]
        inds = [all_inds[i] for i in sample_idx]
        extra_index = [ref_cam] + extra_index  # add target camera index
        inds = [target_index] + inds  # add target frame index
    else:
        extra_index = all_extra_index
        inds = all_inds
    extra_index = torch.as_tensor(extra_index)
    inds = torch.as_tensor(inds)
    return extra_index, inds



def compute_mvseqv2_inds(n_cam, src_inds, target_index, n_srcs, is_train=True):
    # keep the front view and target frame
    src_inds = [target_index] + src_inds   # add target index
    if is_train:
        inds = []
        extra_index = []
        src_inds_ = src_inds[1:]
        random.shuffle(src_inds_)
        src_inds = [src_inds[0]] + src_inds_  # frame index                    
        for src_ind in src_inds:
            inds.append(src_ind)
            extra_index.append(0)
            if len(inds) == n_srcs + 1:
                break
            # ramdom sample other camera index
            sample_idx = random.sample(range(1, n_cam), random.randint(1, n_cam - 1))
            if len(sample_idx) + len(inds) > n_srcs + 1:
                sample_idx = sample_idx[:n_srcs + 1 - len(inds)]
            for sample in sample_idx:
                inds.append(src_ind)
                extra_index.append(sample)
            if len(inds) == n_srcs + 1:
                break
        if len(inds) < n_srcs + 1:
            for src_ind in src_inds * 100:
                inds.append(src_ind)
                extra_index.append(0)
                if len(inds) == n_srcs + 1:
                    break
                # ramdom sample other 5 camera index
                sample_idx = random.sample(range(1, n_cam), n_cam - 1)
                if len(sample_idx) + len(inds) > n_srcs + 1:
                    sample_idx = sample_idx[:n_srcs + 1 - len(inds)]
                for sample in sample_idx:
                    inds.append(src_ind)
                    extra_index.append(sample)
                if len(inds) == n_srcs + 1:
                    break
        inds = inds[:n_srcs + 1]  # in case n_srcs + 1 < len(inds)
        extra_index = extra_index[:n_srcs + 1]
    else:
        all_extra_index = list(range(n_cam)) * len(src_inds)
        all_inds = [[ind] * n_cam for ind in src_inds]
        all_inds = [item for sublist in all_inds for item in sublist]
        extra_index = all_extra_index
        inds = all_inds
    extra_index = torch.as_tensor(extra_index)
    inds = torch.as_tensor(inds)
    return extra_index, inds
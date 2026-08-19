# Run this on a GPU node, e.g.:
#   salloc -c 2 --mem=12gb --gres=gpu:2080ti:1 --time=1-00

import glob
import os

import h5py
import numpy as np

path_datasets = "/dataNAS/people/aagatti/projects/nsm_femur/cache/Sep_12_2023"

list_npz = glob.glob(os.path.join(path_datasets, "*.npz"))

for npz_path in list_npz:
    npz = np.load(npz_path)
    break

with h5py.File(npz_path.replace("npz", "h5"), "w") as hf:
    for key in npz.keys():
        hf.create_dataset(key, data=npz[key])

h5 = h5py.File(npz_path.replace("npz", "h5"), "r")

import torch


def get_pos_neg_indices_h5(
    h5,
    idx,
    samples_per_sign=4250,
):
    perm_pos = torch.randperm(h5[f"pos_idx_{idx}"].shape[0])[:samples_per_sign]
    perm_pos = torch.sort(perm_pos).values
    idx_pos = torch.from_numpy(h5[f"pos_idx_{idx}"][perm_pos])

    perm_neg = torch.randperm(h5[f"neg_idx_{idx}"].shape[0])[:samples_per_sign]
    perm_neg = torch.sort(perm_neg).values
    idx_neg = torch.from_numpy(h5[f"neg_idx_{idx}"][perm_neg])

    indices = torch.cat([idx_pos, idx_neg], dim=0)

    return indices


def load_data_h5(h5, indices):

    indices = torch.unique(indices, sorted=True)

    print(indices.shape)
    print(indices)

    xyz = h5["pts"][()]
    gt_sdf = h5["sdfs"][()]

    data = {
        "xyz": xyz[indices, :],
        "gt_sdf": gt_sdf[indices, :],
    }

    return data


def test(h5, samples_per_sign):

    indices_0 = get_pos_neg_indices_h5(h5=h5, idx=0, samples_per_sign=samples_per_sign)
    indices_1 = get_pos_neg_indices_h5(h5=h5, idx=1, samples_per_sign=samples_per_sign)

    indices = torch.cat([indices_0, indices_1], dim=0)

    data = load_data_h5(h5=h5, indices=indices)


# %timeit test(h5, samples_per_sign=4250)


def unpack_pts(data, pts_name="orig_pts"):
    # get original points...
    pts = []
    pts_arrays = [x for x in data.files if f"{pts_name}_" in x]
    if len(pts_arrays) > 0:
        for pts_idx in range(len(pts_arrays)):
            pts.append(torch.from_numpy(data[f"{pts_name}_{pts_idx}"]))

    return pts


def unpack_numpy_data(
    data_,
    point_cloud=False,
    list_additional_keys=["orig_pts", "new_pts", "pos_idx", "neg_idx", "surf_idx"],
):

    data = {}

    # Get points / xyz coords
    if "pts" in data_:
        data["xyz"] = torch.from_numpy(data_["pts"]).float()
    elif "xyz" in data_:
        data["xyz"] = torch.from_numpy(data_["xyz"]).float()
    else:
        raise ValueError("No pts or xyz in cached file")

    # Get SDFs of the original points
    if "sdfs" in data_:
        data["gt_sdf"] = torch.from_numpy(data_["sdfs"]).float()
    elif "gt_sdf" in data_:
        data["gt_sdf"] = torch.from_numpy(data_["gt_sdf"]).float()
    elif "sdf" in data_:
        data["gt_sdf"] = torch.from_numpy(data_["sdf"]).float()
    else:
        raise ValueError("No sdfs or gt_sdf or sdf in cached file")

    # Get random point cloud surface points... this was used for Diffusion SDFs model
    if point_cloud is True:
        data["point_cloud"] = torch.from_numpy(data_["point_cloud"]).float()

    for key in list_additional_keys:
        key_data = unpack_pts(data_, pts_name=key)
        data[key] = key_data

    return data


def get_pos_neg_indices_npy(
    data,
    idx,
    samples_per_sign=4250,
):
    perm_pos = torch.randperm(data["pos_idx"][idx].shape[0])[:samples_per_sign]
    # perm_pos = torch.sort(perm_pos).values
    idx_pos = data["pos_idx"][idx][perm_pos]

    perm_neg = torch.randperm(data["neg_idx"][idx].shape[0])[:samples_per_sign]
    # perm_neg = torch.sort(perm_neg).values
    idx_neg = data["neg_idx"][idx][perm_neg]

    indices = torch.cat([idx_pos, idx_neg], dim=0)

    return indices


def load_data_npy(data, indices):

    # indices = torch.unique(indices, sorted=True)

    # print(indices.shape)
    # print(indices)

    # xyz = h5['pts'][()]
    # gt_sdf = h5['sdfs'][()]

    data = {
        "xyz": data["xyz"][indices, :],
        "gt_sdf": data["gt_sdf"][indices, :],
    }

    return data


def test_npy(path, samples_per_sign):

    data = unpack_numpy_data(
        data_=np.load(path),
        point_cloud=False,
        # list_additional_keys=['orig_pts', 'new_pts', 'pos_idx', 'neg_idx', 'surf_idx']
        list_additional_keys=["pos_idx", "neg_idx"],
    )

    indices_0 = get_pos_neg_indices_npy(data=data, idx=0, samples_per_sign=samples_per_sign)
    indices_1 = get_pos_neg_indices_npy(data=data, idx=1, samples_per_sign=samples_per_sign)

    indices = torch.cat([indices_0, indices_1], dim=0)

    data = load_data_npy(data=data, indices=indices)


# %timeit -r 10 -n 10 test_npy(npz_path, samples_per_sign=4250)


def test_npy(path, samples_per_sign):

    data = unpack_numpy_data(
        data_=np.load(path),
        point_cloud=False,
        list_additional_keys=["orig_pts", "new_pts", "pos_idx", "neg_idx", "surf_idx"],
    )

    indices_0 = get_pos_neg_indices_npy(data=data, idx=0, samples_per_sign=samples_per_sign)
    indices_1 = get_pos_neg_indices_npy(data=data, idx=1, samples_per_sign=samples_per_sign)

    indices = torch.cat([indices_0, indices_1], dim=0)

    data = load_data_npy(data=data, indices=indices)


# %timeit -r 10 -n 10 test_npy(npz_path, samples_per_sign=4250)


# READING IT ALL INTO DISK IS MUCH FASTER THAN USING H5
# BUT, WE CAN MAKE NUMPY FASTER (READING TO DISK) BY LOADING/PARSING
# LESS OF THE DATA (['pos_idx', 'neg_idx']) vs everything...

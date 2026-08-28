import numpy as np
from scipy.spatial import cKDTree as KDTree


def get_rand_samples(pts1, pts2, num_samples):
    """
    Randomly sample points from two point clouds.

    Args:
    - pts1 (numpy.ndarray): The first point cloud.
    - pts2 (numpy.ndarray): The second point cloud.
    - num_samples (int): The number of points to sample from each point cloud.

    Returns:
    - pts1 (numpy.ndarray): The first point cloud with num_samples points randomly sampled.
    - pts2 (numpy.ndarray): The second point cloud with num_samples points randomly sampled.
    """

    sample1 = np.random.choice(
        pts1.shape[0], size=num_samples, replace=True if pts1.shape[0] < num_samples else False
    )
    pts1 = pts1[sample1, :]

    sample2 = np.random.choice(
        pts2.shape[0], size=num_samples, replace=True if pts2.shape[0] < num_samples else False
    )
    pts2 = pts2[sample2, :]

    return pts1, pts2


def get_pt_cloud_distances(pts1, pts2, num_samples=None):
    """
    Compute the distances between two point clouds.

    Args:
    - pts1 (numpy.ndarray): The first point cloud.
    - pts2 (numpy.ndarray): The second point cloud.
    - num_samples (int, optional): The number of points to randomly sample from each point cloud. If None, all points are used.

    Returns:
    - d1 (numpy.ndarray): The distances from each point in pts2 to its nearest neighbor in pts1
      (length = len(pts2)).
    - d2 (numpy.ndarray): The distances from each point in pts1 to its nearest neighbor in pts2
      (length = len(pts1)).

    Note: compute_assd's denominator (len(pts1) + len(pts2)) is correct regardless of
    which array is which, because the two lengths are only ever summed. Do not "fix"
    the query order to match an older version of this docstring — the arrays swap, the
    sum does not.
    """

    if num_samples is not None:
        pts1, pts2 = get_rand_samples(pts1, pts2, num_samples)

    kd1 = KDTree(pts1)
    kd2 = KDTree(pts2)

    d1, _ = kd1.query(pts2)
    d2, _ = kd2.query(pts1)

    return d1, d2


def compute_assd(
    pts1,
    pts2,
    num_samples=None,
):
    """
    Compute the average symmetric surface distance (ASSD) between two point clouds.

    Args:
    - pts1 (numpy.ndarray): The first point cloud.
    - pts2 (numpy.ndarray): The second point cloud.
    - num_samples (int, optional): The number of points to randomly sample from each point cloud. If None, all points are used.

    Returns:
    - assd (float): The average symmetric surface distance between the two point clouds.
    """
    d1, d2 = get_pt_cloud_distances(pts1, pts2, num_samples)

    return (np.sum(d1) + np.sum(d2)) / (pts1.shape[0] + pts2.shape[0])


def compute_chamfer(
    pts1,
    pts2,
    num_samples=None,
    power=1,
):
    """
    Compute the Chamfer distance between two point clouds.

    Args:
    - pts1 (numpy.ndarray): The first point cloud.
    - pts2 (numpy.ndarray): The second point cloud.
    - num_samples (int, optional): The number of points to randomly sample from each point cloud. If None, all points are used.
    - power (float, optional): The power to raise the distances to before taking the mean. Default is 1.

    Returns:
    - chamfer (float): The Chamfer distance between the two point clouds.
    """

    d1, d2 = get_pt_cloud_distances(pts1, pts2, num_samples)

    return np.mean(d1**power) + np.mean(d2**power)


# Update LR
def adjust_learning_rate(initial_lr, optimizer, iteration, decreased_by, adjust_lr_every):
    lr = initial_lr * ((1 / decreased_by) ** (iteration // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def refuse_unknown_kwargs(kwargs, *, function_name, deprecated=()):
    """Raise on any keyword ``function_name`` neither names nor deprecates.

    Both public entry points take a ``**kwargs`` that reads one key and swallowed the
    rest, so a misspelling among dozens of near-synonymous parameter names ran with the
    intended parameter's default and said nothing at all. One implementation because it
    is one defect: ``reconstruct_mesh`` (58 named parameters, deprecating
    ``batch_size_latent_recon``) and ``reconstruct_latent`` (38, deprecating
    ``max_batch_size``). See ``docs/KNOWN_ISSUES.md`` § History 20.
    """
    unknown = sorted(set(kwargs) - set(deprecated))
    if unknown:
        raise TypeError(
            f"{function_name}() got unexpected keyword arguments: "
            + ", ".join(repr(name) for name in unknown)
        )

"""Fit each class distribution after every bankful of new reliable features."""
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits


def _fit(bank, cls, samples, previous_fits):
    weights = getattr(bank, f"gmm_weights_{cls}")
    means = getattr(bank, f"gmm_means_{cls}")
    covs = getattr(bank, f"gmm_covariances_{cls}")
    with threadpool_limits(limits=1):
        mixture = GaussianMixture(n_components=len(weights), covariance_type="full",
                                  init_params="kmeans", reg_covar=1e-3)
        mixture.fit(samples.detach().cpu().double().numpy())
    fresh = [torch.as_tensor(x, device=samples.device, dtype=samples.dtype)
             for x in (mixture.weights_, mixture.means_, mixture.covariances_)]
    if previous_fits:
        # EM component labels are arbitrary. Match centers before averaging.
        _, order = linear_sum_assignment(torch.cdist(means, fresh[1]).cpu().numpy())
        indices = torch.as_tensor(order, device=samples.device)
        fresh = [old.lerp(new[indices], 1 - bank.gamma)
                 for old, new in zip((weights, means, covs), fresh)]
    fresh[0] = fresh[0] / fresh[0].sum()
    if not all(torch.isfinite(x).all() for x in fresh) or (fresh[0] <= 0).any():
        raise FloatingPointError("Invalid fitted prototype distribution")
    torch.linalg.cholesky(fresh[2])
    for old, new in zip((weights, means, covs), fresh):
        old.copy_(new)


@torch.no_grad()
def collect(bank, cam, features, targets, skull, threshold, capacity, fits):
    """Append one random feature per eligible region and preserve batch overflow."""
    spatial = cam.shape[-2:]
    inside = F.interpolate(skull.float(), size=spatial, mode="nearest").flatten(1).bool()
    activation = cam.detach().flatten(1).relu()
    activation = activation / activation.amax(1, keepdim=True).clamp_min(1e-12)
    positive = targets.reshape(-1, 1).bool()
    regions = (inside & (activation == 0), inside & positive & (activation >= threshold))
    for cls, region in enumerate(regions):
        valid = region.any(1)
        if not valid.any():
            continue
        index = torch.multinomial(region[valid].float(), 1)
        rows = valid.nonzero(as_tuple=True)[0]
        samples = features[rows, index[:, 0]]
        queue = getattr(bank, f"queue{cls}")
        ptr = getattr(bank, f"queue_ptr{cls}")
        start = int(ptr)
        pending = torch.cat((queue[:start], samples))
        while len(pending) >= capacity:
            _fit(bank, cls, pending[:capacity], fits[cls])
            fits[cls] += 1
            pending = pending[capacity:]
        queue[:len(pending)].copy_(pending)
        ptr.fill_(len(pending))

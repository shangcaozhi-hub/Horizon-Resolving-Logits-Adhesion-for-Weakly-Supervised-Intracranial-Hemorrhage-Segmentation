"""Collect weakly labeled patch features and rebuild the archived Gaussian banks."""
import torch
from torch.nn import functional as F
from network.GMM import GaussianMixture


@torch.no_grad()
def collect(bank, cam, features, targets, skull):
    """Use high-CAM patches on positive slices and intracranial patches on negatives."""
    inside = F.interpolate(skull.float(), size=cam.shape[-2:], mode="nearest").flatten(1).bool()
    per_class = [[], []]
    for values, scores, valid, positive in zip(features.detach(), cam.detach().flatten(1),
                                                inside, targets.reshape(-1)):
        indices = valid.nonzero().flatten()
        if not indices.numel():
            continue
        if positive.item() > 0:
            eligible_scores = scores[indices]
            span = eligible_scores.max() - eligible_scores.min()
            if span <= 0:
                continue
            # Relative activation within the skull keeps the rule usable before calibration.
            cutoff = eligible_scores.min() + 0.8 * span
            indices = indices[eligible_scores >= cutoff]
            cls = 1
        else:
            cls = 0
        chosen = torch.randperm(indices.numel(), device=indices.device)[:32]
        per_class[cls].append(values[indices[chosen]])
    for cls, chunks in enumerate(per_class):
        if not chunks:
            continue
        queue = getattr(bank, f"queue{cls}")
        count = getattr(bank, f"queue_ptr{cls}")
        new = torch.cat(chunks)
        recent = torch.cat((queue[:int(count)], new))[-len(queue):]
        queue[:len(recent)].copy_(recent)
        count.fill_(len(recent))


@torch.no_grad()
def rebuild(bank, minimum):
    """Publish both fits together. Keep partial queues for the next epoch."""
    if min(int(bank.queue_ptr0), int(bank.queue_ptr1)) < minimum:
        return False
    fitted = []
    for cls in (0, 1):
        queue = getattr(bank, f"queue{cls}")[:int(getattr(bank, f"queue_ptr{cls}"))]
        components = getattr(bank, f"gmm_weights_{cls}").numel()
        if components == 1:
            mean = queue.mean(0, keepdim=True)
            centered = queue - mean
            cov = (centered.T @ centered / len(queue)).unsqueeze(0)
            cov.diagonal(dim1=-2, dim2=-1).add_(1e-3)
            weights = queue.new_ones(1)
        else:
            indices = torch.randperm(len(queue), device=queue.device)[:components]
            gmm = GaussianMixture(components, queue.shape[1], eps=1e-3,
                                  mu_init=queue[indices].unsqueeze(0)).to(queue.device)
            gmm.fit(queue, n_iter=20)
            weights = gmm.pi.reshape(-1)
            weights = weights / weights.sum()
            mean, cov = gmm.mu[0], gmm.var[0]
        cov = (cov + cov.transpose(-1, -2)) * 0.5
        if not (torch.isfinite(mean).all() and torch.isfinite(cov).all()
                and torch.isfinite(weights).all() and (weights > 0).all()):
            raise FloatingPointError("Invalid prototype fit")
        torch.linalg.cholesky(cov)
        fitted.append((weights, mean, cov))
    for cls, params in enumerate(fitted):
        for name, value in zip(("weights", "means", "covariances"), params):
            getattr(bank, f"gmm_{name}_{cls}").copy_(value)
        getattr(bank, f"queue_ptr{cls}").zero_()
    return True

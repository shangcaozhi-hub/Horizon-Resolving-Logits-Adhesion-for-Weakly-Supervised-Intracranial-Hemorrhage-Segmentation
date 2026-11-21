import torch
from torch import nn
import torch.nn.functional as F
import random
from network.ppc_loss import sample_ppc
from models import score_and_cam


def _forward_view(model, image, view=None):
    # Each view samples independently from the current fitted bank.
    scores, cam, features, foreground, above = model.cam_phase(image) if view is None else view
    logits, _, _, _ = model.proto_phase(features, cam)
    logits = logits[:, :, :model.num_prototype]
    similarity = logits.amax(-1).reshape(cam.size(0), *cam.shape[-2:], 2).permute(0, 3, 1, 2)
    proto_mask = similarity[:, 1:2] > similarity[:, :1]
    if isinstance(scores, (list, tuple)):
        scores = scores[0]
    return scores, cam, logits, similarity, proto_mask, foreground, above


def _normalize_cam(cam):
    positive = F.relu(cam)
    max = (F.adaptive_max_pool2d(positive, (1, 1)) + 1e-5).detach()
    return positive / max


def _refined_label(proto_mask, fore_mask, above_mask):
    uncertain = proto_mask | above_mask
    label = torch.zeros_like(proto_mask, dtype=torch.long)
    label[uncertain] = -1
    label[proto_mask & fore_mask] = 1
    return label, uncertain


def compute_horizon_losses(model, image, label, skull_mask=None, flip_dims=None, first_view=None):
    """CLS + 0.1*PPC + 0.1*CVC + C2P, including view-1 PPC anchoring."""
    if flip_dims is None:
        horizontal = random.random() < 0.5
        vertical = random.random() < 0.5
        if not horizontal:
            vertical = True
        flip_dims = tuple(dim for dim, enabled in ((-1, horizontal), (-2, vertical)) if enabled)
    view2 = torch.flip(image, flip_dims)
    scores1, cam1, logits1, sim1, proto1, fore1, above1 = _forward_view(
        model, image, first_view)
    scores2, cam2, _, sim2, proto2, fore2, above2 = _forward_view(model, view2)
    label = label.reshape(-1).float()
    cls_loss = 0.5 * (
        F.binary_cross_entropy_with_logits(scores1.reshape(-1).float(), label) +
        F.binary_cross_entropy_with_logits(scores2.reshape(-1).float(), label))

    # Negative images contribute background pixels in both views.
    positive_images = (label != 0).reshape(-1, 1, 1, 1)
    proto1 = proto1 & positive_images
    above1 = above1 & positive_images
    proto2 = proto2 & positive_images
    above2 = above2 & positive_images
    options = dict(getattr(model, "ppc_options", {}))
    options["num_components"] = model.n_components
    if options.get("mode", "hard") == "weighted":
        # Label-free weighted PPC evaluates both class directions at every pixel.
        valid_mask = None
        if skull_mask is not None:
            valid_mask = F.interpolate(
                skull_mask.float(), size=cam1.shape[-2:], mode="nearest") > 0
        ppc_loss = sample_ppc(
            None, logits1, cam_logits=cam1, valid_mask=valid_mask, **options)
    else:
        # Both pseudo-label maps supervise view-1 logits, as in the original engine.
        refined1, _ = _refined_label(proto1, fore1, above1)
        refined2, _ = _refined_label(torch.flip(proto2, flip_dims),
                                    torch.flip(fore2, flip_dims),
                                    torch.flip(above2, flip_dims))
        ppc_loss = 0.5 * (
            sample_ppc(refined1, logits1, **options) +
            sample_ppc(refined2, logits1, **options))

    background = sim1[:, :1] > sim1[:, 1:2]
    has_background = background.flatten(2).any(dim=2)[:, :, None, None]
    bias = cam1.masked_fill(~background, -torch.inf).amax(dim=(2, 3), keepdim=True)
    bias = torch.where(has_background, bias, torch.zeros_like(bias))
    def relative_activation(reference):
        shifted = (cam1 - reference).relu()
        return shifted / shifted.amax(dim=(2, 3), keepdim=True).clamp_min(1e-5)

    biased1 = relative_activation(bias)

    aligned_sim2 = torch.flip(sim2, flip_dims)

    proto_cam1 = _normalize_cam(sim1[:, 1:2] - sim1[:, :1])
    proto_cam2 = _normalize_cam(aligned_sim2[:, 1:2] - aligned_sim2[:, :1])
    cvc_loss = (proto_cam2 - proto_cam1).abs().mean()
    positive_mask = (positive_images & has_background).reshape(-1)

    # normal forward: bias participates in gradient
    q_live = biased1[positive_mask]
    # same forward value, but bias gradient is blocked
    biased1_stop_bias = relative_activation(bias.detach())
    q_stop = biased1_stop_bias[positive_mask]
    # prototype supervision
    p = proto_cam1[positive_mask].detach()
    eps = 1e-6

    # mask before clamp
    # valid_mask = ((p > 0) | (q_live.detach() > 0)).to(q_live.dtype)

    p = p.clamp(eps, 1.0 - eps)
    q_live = q_live.clamp(eps, 1.0 - eps)
    q_stop = q_stop.clamp(eps, 1.0 - eps)

    kl_map = (p * (torch.log(p) - torch.log(q_live)) + (1.0 - p) * (torch.log1p(-p) - torch.log1p(-q_stop)))

    c2p_loss = kl_map.mean() if kl_map.numel() else cam1.new_zeros(()) # (kl_map * valid_mask).sum() / (valid_mask.sum() + eps)
    loss = cls_loss + 0.1 * ppc_loss + 0.1 * cvc_loss + 1 * c2p_loss
    return dict(loss=loss, cls_loss=cls_loss, ppc_loss=ppc_loss,
                cvc_loss=cvc_loss, c2p_loss=c2p_loss)


def batch_loss(name, model, image, label, skull_mask):
    """Return a differentiable loss for a batch from RSNAClsDataset."""
    if name in ("horizon_spol", "horizon_swin"):
        return compute_horizon_losses(model, image, label, skull_mask)["loss"]
    score, _ = score_and_cam(model, image)
    return F.binary_cross_entropy_with_logits(score, label.reshape(-1).float())

"""Train the four hemorrhage models on RSNA PNG slices."""
import argparse
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataloader.datasets import RSNAClsDataset
from models import MODELS, build_model, load_model
from network.bank_training import collect, collect_spol
from training import batch_loss, compute_horizon_losses


def save_checkpoint(path, state):
    # Publish only a fully written checkpoint. Concurrent runs cannot clobber it.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
    try:
        torch.save(state, tmp)
        if os.name == "nt":
            # Windows rename publishes atomically and fails if path already exists.
            os.rename(tmp, path)
        else:
            os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def stage_learning_rate(epoch_index, duration, peak, floor):
    """Two linear warm-up epochs, then a cosine from peak to floor."""
    warmup_epochs = 2
    if epoch_index < warmup_epochs:
        return floor + (peak - floor) * epoch_index / warmup_epochs
    progress = min((epoch_index - warmup_epochs) / (duration - warmup_epochs - 1), 1.)
    return floor + (peak - floor) * (1. + math.cos(math.pi * progress)) / 2.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--lists", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--checkpoint", type=Path, help="Initialize from trusted project weights")
    start.add_argument("--resume", type=Path, help="Resume a checkpoint from this training entry")
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs including resumed epochs")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float, help="Joint-training AdamW learning rate (default 5e-5)")
    parser.add_argument("--bank-min", type=int, help="Feature count threshold for prototype fits (default 1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    saved = None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("method") != args.model or "optimizer" not in saved:
            parser.error("--resume requires a training checkpoint from the selected model")
        model = build_model(args.model, saved["models"]).to(device)
    elif args.checkpoint:
        model = load_model(args.model, args.checkpoint, device)
    else:
        model = build_model(args.model).to(device)
    model.train()
    horizon = args.model.startswith("horizon_")
    online = saved["online"] if saved is not None else horizon and not args.checkpoint
    collect_features = collect_spol if args.model == "horizon_spol" else collect
    config = saved.get("schedule", {}) if saved is not None else {}
    warmup = config.get("classification_epochs", 10 if online else 0)
    joint_lr = args.lr if args.lr is not None else config.get("joint_lr", 5e-5)
    if args.batch_size is None:
        args.batch_size = config.get("batch_size", 48)
    if args.epochs is None:
        args.epochs = warmup + 20
    minimum = args.bank_min if args.bank_min is not None else config.get("capacity", 1000)
    if args.batch_size < 2 or args.epochs < 1 or joint_lr <= 0:
        parser.error("Use batch size >= 2, positive epochs and a positive learning rate.")
    fits, steps, first_epoch = [0, 0], 0, 1
    if saved is not None and len(saved["optimizer"]["param_groups"]) == 1:
        parameters = model.parameters()
    else:
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                (no_decay if param.ndim == 1 or name.endswith(".bias") else decay).append(param)
        parameters = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(parameters, lr=joint_lr, weight_decay=0.05)
    if saved is not None:
        fits = saved.get("class_fits", [saved["bank_fits"]] * 2)
        steps = saved["steps"]
        first_epoch = saved["epoch"] + 1
        if "schedule" not in saved:
            # Earlier releases entered joint training as soon as both banks fitted.
            warmup = 0 if min(fits) else warmup
            if args.bank_min is None:
                minimum = saved["bank_min"]
        if online and args.model == "horizon_spol" and not config.get("rolling_spol_bank"):
            bank = model.PrototypeSupportBank
            for cls in (0, 1):
                queue = getattr(bank, f"queue{cls}")
                count = int(getattr(bank, f"queue_ptr{cls}"))
                queue[:count].copy_(queue[:count].flip(0))
                queue[count:].zero_()
        optimizer.load_state_dict(saved["optimizer"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        del saved
    if first_epoch > args.epochs:
        parser.error("--epochs must include at least one new epoch")
    if online and not model.n_components < minimum <= model.PrototypeSupportBank.bank_size:
        parser.error("--bank-min must exceed the background component count and fit in the queue")
    dataset = RSNAClsDataset(args.lists, args.images, train=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    if len(loader) == 0:
        parser.error("Not enough slices for one complete batch")
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(first_epoch, args.epochs + 1):
        destination = args.output / f"epoch-{epoch:03d}.pth"
        if destination.exists():
            raise FileExistsError(destination)
        if epoch <= warmup:
            lr = stage_learning_rate(epoch - 1, warmup, 5e-4, 5e-5)
        else:
            lr = stage_learning_rate(epoch - warmup - 1, 20, joint_lr, min(1e-5, joint_lr))
        for group in optimizer.param_groups:
            group["lr"] = lr
        total, phase = 0., "classification"
        for image, target, skull in loader:
            image, target, skull = image.to(device), target.to(device), skull.to(device)
            optimizer.zero_grad(set_to_none=True)
            if online:
                view = model.cam_phase(image)
                scores, cam, features, _, _ = view
                collect_features(model.PrototypeSupportBank, cam, features, target, skull,
                                 model.threshold, minimum, fits)
                if epoch > warmup and not min(fits):
                    raise RuntimeError("Joint training requires both fitted banks. Check CAM regions, "
                                       "slice labels and the number of eligible training examples.")
                if epoch > warmup:
                    loss = compute_horizon_losses(model, image, target, skull, first_view=view)["loss"]
                    phase = "horizon"
                else:
                    score = scores[0] if isinstance(scores, (list, tuple)) else scores
                    loss = F.binary_cross_entropy_with_logits(score.reshape(-1), target.reshape(-1).float())
                    phase = "warmup"
            else:
                loss = batch_loss(args.model, model, image, target, skull)
                if horizon:
                    phase = "horizon"
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {steps + 1}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            total += loss.item()
        save_checkpoint(destination, dict(
            method=args.model, models=model.state_dict(), optimizer=optimizer.state_dict(),
            epoch=epoch, steps=steps, online=online, bank_fits=min(fits), bank_min=minimum, class_fits=fits,
            schedule=dict(classification_epochs=warmup, joint_lr=joint_lr,
                          batch_size=args.batch_size, capacity=minimum,
                          rolling_spol_bank=online and args.model == "horizon_spol"),
            torch_rng=torch.get_rng_state(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None))
        print(f"epoch={epoch} steps={steps} lr={lr:g} phase={phase} mean_loss={total/len(loader):.5f} "
              f"last_grad_norm={norm.item():.5g} bank_fits={fits} file={destination}", flush=True)
    if online and not min(fits):
        print("Warm-up only: the queues still need foreground and background samples. "
              "Resume for more epochs; the checkpoint retains the collected features.")


if __name__ == "__main__":
    main()

"""Train the four hemorrhage models on RSNA PNG slices."""
import argparse
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
from network.bank_training import collect, rebuild
from training import batch_loss, compute_horizon_losses


def save_checkpoint(path, state):
    # Publish only a fully written checkpoint. Concurrent runs cannot clobber it.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
    try:
        torch.save(state, tmp)
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--lists", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--checkpoint", type=Path, help="Initialize from trusted project weights")
    start.add_argument("--resume", type=Path, help="Resume a checkpoint from this training entry")
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5, help="Initial AdamW learning rate")
    parser.add_argument("--bank-min", type=int, help="Minimum queued features per class to refit (default 256)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 2 or args.epochs < 1 or args.lr <= 0:
        parser.error("Use batch size >= 2, positive epochs and a positive learning rate.")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    horizon = args.model.startswith("horizon_")
    online = horizon and not args.checkpoint
    fits, steps, first_epoch = 0, 0, 1
    minimum = args.bank_min if args.bank_min is not None else 256
    if saved is not None:
        online, fits, steps = saved["online"], saved["bank_fits"], saved["steps"]
        first_epoch = saved["epoch"] + 1
        if args.bank_min is None:
            minimum = saved["bank_min"]
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
        total, phase = 0., "classification"
        for image, target, skull in loader:
            image, target, skull = image.to(device), target.to(device), skull.to(device)
            optimizer.zero_grad(set_to_none=True)
            if online:
                view = model.cam_phase(image)
                scores, cam, features, _, _ = view
                collect(model.PrototypeSupportBank, cam, features, target, skull)
                if fits:
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
        if online:
            fits += int(rebuild(model.PrototypeSupportBank, minimum))
        save_checkpoint(destination, dict(
            method=args.model, models=model.state_dict(), optimizer=optimizer.state_dict(),
            epoch=epoch, steps=steps, online=online, bank_fits=fits, bank_min=minimum,
            torch_rng=torch.get_rng_state(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None))
        print(f"epoch={epoch} steps={steps} phase={phase} mean_loss={total/len(loader):.5f} "
              f"last_grad_norm={norm.item():.5g} bank_fits={fits} file={destination}", flush=True)
    if online and fits == 0:
        print("Warm-up only: the queues still need foreground and background samples. "
              "Resume for more epochs; the checkpoint retains the collected features.")


if __name__ == "__main__":
    main()

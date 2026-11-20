"""Run the native slice loader and export unthresholded model outputs."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from dataloader.fully_datasets import insDataSet, bhsdDataSet
from models import MODELS, load_model, score_and_cam


def save_jpg(stream, image, mask, cam, case, score):
    """Render aligned CT, reference mask and positive CAM in the input frame."""
    gray = np.rint(image[0].clamp(0, 1).numpy() * 255).astype(np.uint8)
    ct = np.repeat(gray[..., None], 3, axis=-1)
    gt = ct.copy()
    lesion = mask.numpy() > 0
    gt[lesion] = 0.45 * gt[lesion] + 0.55 * np.array([20, 220, 100])
    h, w = gray.shape
    activation = cam[0].clamp_min(0)
    activation = activation / activation.max().clamp_min(1e-8)
    heat = Image.fromarray(activation.numpy()).resize((w, h), Image.Resampling.BILINEAR)
    strength = np.asarray(heat)[..., None]
    color = np.concatenate((np.ones_like(strength) * 255, 210 * strength, 0 * strength), axis=2)
    alpha = 0.8 * strength
    overlay = np.rint((1 - alpha) * ct + alpha * color).astype(np.uint8)
    canvas = Image.new("RGB", (3 * w, h + 30), (23, 27, 31))
    draw = ImageDraw.Draw(canvas)
    panels = [(ct, f"CT  {Path(case).stem}"), (gt, "Ground truth"), (overlay, f"CAM   logit {score:.2f}")]
    for index, (pixels, title) in enumerate(panels):
        canvas.paste(Image.fromarray(pixels), (index * w, 30))
        draw.text((index * w + 8, 9), title, fill="white")
    canvas.save(stream, format="JPEG", quality=95, subsampling=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--list", required=True, help="Slice list relative to data root")
    parser.add_argument("--dataset", choices=("instance", "bhsd"), default="instance")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--format", choices=("jpg", "npz"), default="jpg")
    args = parser.parse_args()
    model = load_model(args.model, args.checkpoint, args.device)
    dataset_type = insDataSet if args.dataset == "instance" else bhsdDataSet
    dataset = dataset_type(str(args.data_root), "/" + args.list.lstrip("/"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    args.output.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(args.device)
            if args.dataset == "bhsd":
                images = torch.rot90(images, dims=(-2, -1))
            scores, cams = score_and_cam(model, images)
            if args.dataset == "bhsd":
                cams = torch.rot90(cams, k=-1, dims=(-2, -1))
            for i, (case, score, cam) in enumerate(zip(batch["case"], scores.cpu(), cams.cpu())):
                destination = args.output / f"{Path(case).stem}.{args.format}"
                with destination.open("xb") as stream:
                    if args.format == "jpg":
                        save_jpg(stream, batch["image"][i], batch["mask"][i], cam, case, score.item())
                    else:
                        np.savez_compressed(stream, case=case, score=score.numpy(), cam=cam.numpy())


if __name__ == "__main__":
    main()

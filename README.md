# Horizon

**Resolving Logits Adhesion for Weakly Supervised Intracranial Hemorrhage Segmentation**

Caozhi Shang, Xian Lin, Junjie Shi, Zengqiang Yan, Xianbo Deng, and Li Yu

Horizon learns to localize intracranial hemorrhage on non-contrast head CT using a binary label for each training slice. A classifier can correctly identify a positive slice while assigning similar scores to a subtle bleed and nearby normal tissue. The manuscript calls this overlap between their pixel-wise logit distributions **logits adhesion**. Horizon uses the feature distributions of hemorrhage and normal tissue to teach the classifier which local responses to raise or suppress.

This PyTorch release includes Horizon with SPOL and Swin-MIL backbones, both standalone baselines, an RSNA training entry point, and slice-level inference for INSTANCE and BHSD.

## How Horizon works

Horizon fits a Gaussian mixture model (GMM) to each class and samples prototypes that represent hemorrhage and background tissue. Their similarities to each image location form a prototype activation map (PAM). During training, the PAM guides the classifier across two views of the same slice. At inference, the classifier activation map localizes hemorrhage.

![Figure 3. Two-view training framework and relative activation.](docs/figures/figure-3-with-relative-activation.png)

The relative activation map (RAM) uses the strongest background logit as a reference. A lesion pixel can therefore have positive relative activation even when its raw logit is negative.

![Figure 4. Gradient flow during logit refinement.](docs/figures/figure-4-logit-gradients.png)

![Comparison of erasing, CAM calibration, pseudo-label fitting and Horizon.](docs/figures/horizon-comparison.png)

## Results reported in the manuscript

Table 2 reports patient-volume Dice as mean ± standard deviation over four independent runs. The Logits Adhesion Index (LAI) measures the overlap of hemorrhage and normal-tissue logit densities. Lower LAI indicates better separation. The segmentation threshold was selected on held-out INSTANCE validation data and then fixed for testing.

| Model | CLI name | INSTANCE Dice ↑ (%) | INSTANCE LAI ↓ (%) | BHSD Dice ↑ (%) | BHSD LAI ↓ (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| SPOL | `spol` | 49.90 ± 3.33 | 11.58 | 29.68 ± 2.37 | 27.86 |
| Horizon (SPOL) | `horizon_spol` | 59.56 ± 1.36 | 7.52 | 39.55 ± 0.59 | 23.38 |
| Swin-MIL | `swin_mil` | 56.53 ± 2.76 | 8.48 | 35.80 ± 1.58 | 17.93 |
| Horizon (Swin-MIL) | `horizon_swin` | 63.79 ± 0.32 | 4.26 | 43.43 ± 0.18 | 14.38 |

## Horizon-SPOL weights

The release includes the Horizon-SPOL weights at `weights/horizon_spol_weights.pth` (100.67 MiB). This PyTorch state dictionary contains the learned parameters and fitted prototype bank.

## Test Horizon-SPOL on ten positive INSTANCE slices

1. From the repository root, install the dependencies in a Python 3.9 environment.

   ```sh
   python -m pip install -r requirements.txt
   ```

2. The [joytest manifest](data/INSTANCE/joytest.txt) lists ten hemorrhage-positive INSTANCE slices from ten patients. The [bundled HDF5 inputs](data/INSTANCE/data/slices/) each contain the CT image and its reference mask (`image` and `label`). The command below reads them directly.

3. Run inference. Use a new output directory for each run.

   ```sh
   python predict.py --model horizon_spol --checkpoint weights/horizon_spol_weights.pth --data-root data/INSTANCE --list joytest.txt --format jpg --output outputs/joytest
   ```

4. Open `outputs/joytest/`. The [published joytest results](results/joytest/) include the ten JPGs and a CSV of slice-level predictions. A complete run produces ten JPGs whose names match the slice IDs. Each image shows the input CT, the reference lesion mask in green and the classifier CAM with the slice logit. The heatmap is normalized to the largest positive CAM response on that slice. This all-positive example checks inference and image export; use the full evaluation split for segmentation metrics.

### Other inference inputs

For a full INSTANCE list, replace `joytest.txt` with your list relative to the data root, for example `val_slices.txt`. For BHSD, pass `--dataset bhsd` and a list of case IDs. Each ID maps to `label_192/vol/<id>_vol.npy` and `label_192/seg/<id>_seg.npy` under the BHSD data root. The loader resizes slices to 240 × 240 and repeats the grayscale channel three times. BHSD inference rotates the CAM back into the mask coordinate frame before export. Use `--format npz` for raw CAMs, case IDs and classification logits.

## Train on RSNA

Keep all slices from a patient in the same split. Clip CT intensities to 0-100 HU, apply min-max normalization and save one grayscale PNG per slice in `data/RSNA/pngs/`. In `data/RSNA/lists/train.csv`, the first two columns give the image stem and the slice-level label (1 for any hemorrhage, 0 otherwise).

```csv
Image,any
ID_001,1
ID_002,0
```

Run the following command from the repository root. The loader augments each training slice and converts it to a 240 x 240 three-channel input.

```sh
python train.py --model horizon_spol --images data/RSNA/pngs --lists data/RSNA/lists --output runs/horizon_spol
```

Horizon trains from scratch with AdamW and a batch size of 48. Each stage starts with two warm-up epochs followed by cosine decay.

| Epochs | Objective | Peak learning rate | Minimum learning rate |
| --- | --- | --- | --- |
| 1-10 | Slice classification | 5e-4 | 5e-5 |
| 11-30 | Classification, prototype contrast, cross-view consistency and self-distillation | 5e-5 | 1e-5 |

During the first ten epochs, the classifier learns to detect hemorrhage from slice labels while the model builds its prototype bank. Joint training then uses the augmented slice and a flipped copy as its two views. Their prototype activation maps guide the classifier toward consistent lesion responses across views.

For the Swin backbone, use `--model horizon_swin` and a separate output directory. The trainer saves a checkpoint after each epoch. To continue from epoch 6, rerun the command above with `--resume runs/horizon_spol/epoch-005.pth`. This restores the model, optimizer and prototype bank.

"""Load one of the four hemorrhage models and expose its primary classification logit and CAM."""
from pathlib import Path
import torch
from network.horizon_spol import Plug_play
from network.horizon_swin import PrototypeSwinTransformer
from network.spol import Net
from network.swin_mil import SwinMIL

MODELS = ("horizon_spol", "horizon_swin", "swin_mil", "spol")


def load_model(name, checkpoint, device="cpu"):
    """Load a trusted project checkpoint. Preserve the archived state-dict names."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Available models: {MODELS}")
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if saved.get("online") and saved.get("bank_fits") == 0:
        raise ValueError("Prototype bank is still warming up. Use --resume.")
    return build_model(name, saved.get("models", saved)).to(device).eval()


def build_model(name, state=None):
    """Construct from random initialization or strictly restore matching tensors."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}")
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    bank = "PrototypeSupportBank."
    shape_args = {}
    if name.startswith("horizon_") and state is not None:
        shape_args = dict(components=state[bank + "gmm_means_0"].shape[0],
                          foreground_components=state[bank + "gmm_means_1"].shape[0])
    if name == "horizon_spol":
        model = Plug_play(num_cls=1, num_prototype=50, gamma=0.9, threshold=0.8, **shape_args)
    elif name == "horizon_swin" and state is None:
        model = PrototypeSwinTransformer(auxiliary_head=False)
    elif name == "horizon_swin":
        bank = "PrototypeSupportBank."
        if bank + "queue0" not in state or bank + "gmm_means_0" not in state:
            raise ValueError("Horizon-Swin requires the prototype bank in the checkpoint.")
        model = PrototypeSwinTransformer(
            auxiliary_head="classifier.weight" in state,
            bank_size=state[bank + "queue0"].shape[0],
            num_prototype=100 if shape_args["foreground_components"] == 1 else 50,
            **shape_args)
    elif name == "swin_mil":
        model = SwinMIL(num_classes=1)
    else:
        model = Net(num_cls=1)
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def score_and_cam(model, image):
    """Return [B] image logits and [B, 1, H, W] raw foreground CAMs."""
    scores, cam, *_ = model(image)
    if isinstance(scores, (list, tuple)):
        scores = scores[0]
    return scores.reshape(image.shape[0]), cam

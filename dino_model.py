import torch
from pathlib import Path

from config import DINO_CHECKPOINT_PATH
from dinov3.models.vision_transformer import vit_base


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dino():
    print("Loading DINOv3 ViT-B/16 checkpoint...")
    ckpt_path = Path(DINO_CHECKPOINT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DINO checkpoint not found: {ckpt_path}")

    model = vit_base(16)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    if "teacher" in ckpt:
        ckpt = ckpt["teacher"]

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    return model.to(device).eval()


@torch.no_grad()
def get_embedding(model, img_tensor):
    """
    img_tensor: (3, H, W)
    returns: (D,)
    """
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    img_tensor = img_tensor.to(device)
    outputs = model(img_tensor)
    return outputs.squeeze(0).detach().cpu()

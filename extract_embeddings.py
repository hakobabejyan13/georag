import glob
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

from config import DATASET_DIR, EMBEDDINGS_PATH, EMBEDDING_IDS_PATH
from dataset_loader import load_hdf5
from dino_model import get_embedding, load_dino
from preprocess import extract_rgb, normalize_percentile, to_pil


def main():
    print("Initializing manual DINOv3-B/16 model...")
    model = load_dino()
    model.eval()

    paths = sorted(glob.glob(str(DATASET_DIR / "*.hdf5")))
    print("Found", len(paths), "files.")

    embeddings = []
    embedding_ids = []
    to_tensor = transforms.ToTensor()

    for path in tqdm(paths):
        sample_id = Path(path).stem
        arr13 = load_hdf5(path)
        rgb = extract_rgb(arr13)
        rgb_norm = normalize_percentile(rgb)
        pil_img = to_pil(rgb_norm)
        img_tensor = to_tensor(pil_img)

        with torch.no_grad():
            emb = get_embedding(model, img_tensor)

        emb = emb.squeeze()
        embeddings.append(emb if emb.ndim == 1 else emb.flatten())
        embedding_ids.append(sample_id)

    embeddings = np.stack(embeddings).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    output_path = Path(EMBEDDINGS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    with Path(EMBEDDING_IDS_PATH).open("w", encoding="utf-8") as f:
        json.dump(embedding_ids, f)

    print(f"DONE: {output_path} saved")
    print(f"DONE: {EMBEDDING_IDS_PATH} saved")
    print("Shape:", embeddings.shape)
    print("Dtype:", embeddings.dtype)


if __name__ == "__main__":
    main()

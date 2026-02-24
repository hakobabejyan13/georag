from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "m-bigearthnet"

# Core project artifacts
EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings2.npy"
EMBEDDING_IDS_PATH = PROJECT_ROOT / "embeddings2_ids.json"
INDEX_PATH = PROJECT_ROOT / "georag.index"
LABELS_PATH = DATASET_DIR / "label_stats.json"
DINO_CHECKPOINT_PATH = PROJECT_ROOT / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
LABEL_NAMES_PATH = PROJECT_ROOT / "label_names.json"
CLASS_DESCRIPTIONS_PATH = PROJECT_ROOT / "class_descriptions.json"

# FAISS / RAG settings
USE_COSINE = True
TOP_K = 5
USE_LABEL_NAMES = False

# LLM settings (for HF models or OpenAI)
ENABLE_HF_GENERATION = False
LLM_MODEL_NAME = "distilgpt2"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.2
DEVICE = 0

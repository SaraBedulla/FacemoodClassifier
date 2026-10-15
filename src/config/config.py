"""Project-wide paths and constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
RUNS_DIR = RESULTS_DIR / "runs"

SEED = 42

# Native image format each dataset is stored in after preparation.
DATASETS = {
    "rafdb": {"img_size": 100, "channels": 3},
    "fer2013": {"img_size": 48, "channels": 1},
}
DEFAULT_DATASET = "fer2013"


def processed_path(dataset):
    return PROCESSED_DIR / f"{dataset}_splits.npz"

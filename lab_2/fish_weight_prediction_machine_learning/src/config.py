from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "fish_participant.csv"
DATA_SPLIT = PROJECT_DIR / "src" / "split_indices.json"
TARGET_COLUMN = "Weight"
NUMERIC_COLUMNS = ["Length3", "Height", "Width"]
CATEGORICAL_COLUMNS = ["Species"]
GRAF_PATH = PROJECT_DIR / "graf"
SEED = 42
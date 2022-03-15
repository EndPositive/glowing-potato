from pathlib import Path

RESOURCES_DIR = Path(__file__).parents[1].joinpath("resources")
EDGE_DIR = RESOURCES_DIR.joinpath("edge")
DATASET_DIR = RESOURCES_DIR.joinpath("dataset")
OUTPUT_DIR = RESOURCES_DIR.joinpath("out")
INPUT_DIR_PRECOMPUTED = RESOURCES_DIR.joinpath("precomputed", "in")
OUTPUT_DIR_PRECOMPUTED = RESOURCES_DIR.joinpath("precomputed", "out")

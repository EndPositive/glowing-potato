from pathlib import Path

RESOURCES_DIR = Path(__file__).parents[1].joinpath("resources")
EDGE_DIR = RESOURCES_DIR.joinpath("edge")
DATASET_DIR = RESOURCES_DIR.joinpath("dataset")
DATASET_TEST_DIR = RESOURCES_DIR.joinpath("test_dataset/open-images-v6/test/data")
OUTPUT_DIR = RESOURCES_DIR.joinpath("out")
OUTPUT_DIR_2 = RESOURCES_DIR.joinpath("out2")
INPUT_DIR_PRECOMPUTED = RESOURCES_DIR.joinpath("precomputed", "in")
OUTPUT_DIR_PRECOMPUTED = RESOURCES_DIR.joinpath("precomputed", "out")

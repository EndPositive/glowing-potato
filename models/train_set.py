import gc
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from torchvision.datasets import VisionDataset

from preprocessing import DATASET_DIR
from preprocessing.watermarker import Watermarker


class ChunkedWatermarkedSet(VisionDataset):
    dataset_paths: List[Path] = []

    def __init__(self, transform: Optional[Callable]) -> None:
        super(ChunkedWatermarkedSet, self).__init__(DATASET_DIR, transform=transform)

        self.dataset_paths = [
            jpg_path_in_dataset
            for jpg_path_in_dataset in DATASET_DIR.glob("*.jpg")
            if not Watermarker.load_image(jpg_path_in_dataset)[1]
        ]

        gc.collect()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        watermarker = Watermarker(self.dataset_paths[index])
        original = watermarker.image.copy()
        watermarker.add_default()
        watermarked = self.transform(watermarker.image)
        original = self.transform(original)
        return watermarked, original

    def __len__(self) -> int:
        return len(self.dataset_paths)

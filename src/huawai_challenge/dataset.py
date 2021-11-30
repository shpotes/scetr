from typing import Any, Tuple
from pathlib import Path

from torchvision import datasets

class CocoDetection(datasets.CocoDetection):
    """
    Regular COCO dataset but with SCTR feature extractor instead of transform function
    """
    def __init__(self, img_dir, ann_file, feature_extractor):
        super().__init__(img_dir, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, index: int) -> Tuple[Any, Any]: # TODO: improve type hints
        img, annon = super().__getitem__(index)
        
        image_id = self.ids[index]
        target = {"image_id": image_id, "annotations": annon}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["features"].squeeze() # Remove batch dimension
        target = encoding["label"][0]

        return pixel_values, target
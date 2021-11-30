from typing import Any, Dict
from itertools import chain
import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImagePath
from tqdm import tqdm

def get_info() -> Dict[str, str]:
    return {
        "year": "2021",
        "version": "0.1",
        "description": "Data for Scene text detection hackathon",
        "contributor": "Huawei - ultrahack",
        "url": "https://ultrahack.org/scene-text-detection",
        "date_created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    }

def get_image_info(img: Path) -> Dict[str, Any]:
    _id = img.stem
    fname = img.name
    with Image.open(img) as buf:
        width, height = buf.size

    return {
        "id": int(_id),
        "file_name": fname,
        "height": height,
        "width": width,
        "date_captured": None
    }

def get_annon(line: str, image_id: int) -> Dict[str, Any]:
    pts = np.array(line.split(","), dtype=np.int32).reshape(-1, 2)

    xmin = int(pts[:, 0].min())
    xmax = int(pts[:, 0].max())
    ymin = int(pts[:, 1].min())
    ymax = int(pts[:, 1].max())

    area = (xmax - xmin) * (ymax - ymin)
    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
    rec = cv2.minAreaRect(pts)
    segmentation = pts.reshape(-1).tolist()
    
    return {
        "image_id": int(image_id),
        "bbox": bbox,
        "segmentation": segmentation,
        "area": area,
        "min_rectangle": rec
    }

def get_image_annotation_without_id(annon: Path) -> Dict[str, Any]:
    image_id = annon.stem
    with open(annon) as buf:
        content = buf.read()
    
    return [get_annon(line, image_id) for line in content.split("\n") if line]

def drive_to_coco_format(image_dir: Path, annon_dir: Path) -> Dict[str, Dict[str, Any]]:
    availabel_images = {img.stem for img in image_dir.iterdir() if img.is_file()} 
    availabel_labels = {annon.stem for annon in annon_dir.iterdir() if annon.is_file()}

    total_imgs = len(availabel_images)
    images = [
        get_image_info(img) for img in tqdm(image_dir.iterdir(), total=total_imgs) 
        if img.stem in availabel_labels
    ]

    total_annons = len(availabel_labels)
    annotations = list(chain.from_iterable(
        get_image_annotation_without_id(annon) for annon in tqdm(annon_dir.iterdir(), total=total_annons)
        if annon.stem in availabel_images
    ))

    for _id, annon in enumerate(annotations):
        annon["id"] = _id

    return {
        "info": get_info(),
        "images": images,
        "annotations": annotations
    }

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Convert drive format to coco format")
    argparser.add_argument("--image_dir", type=str, help="Path to image directory")
    argparser.add_argument("--annon_dir", type=str, help="Path to annotation directory")
    argparser.add_argument("--output", type=str, help="Path to output file")

    args = argparser.parse_args()
    image_dir = Path(args.image_dir)
    annon_dir = Path(args.annon_dir)

    coco_format = drive_to_coco_format(image_dir, annon_dir)
    with open(args.output, "w") as buf:
        json.dump(coco_format, buf)
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path

import cv2
from torch.utils.data.dataset import Dataset


class CocoDetectionDataset(Dataset):
    def __init__(self, coco_dict, images_path):
        self.coco_dict = coco_dict
        self.image2annotation_map = collect_annotations(coco_dict)
        self.images_path = Path(images_path)

    def __len__(self):
        return len(self.coco_dict["images"])

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        image_dict = self.coco_dict["images"][idx]
        img = cv2.imread(str(self.images_path / image_dict["file_name"]))
        img_annotations = self.image2annotation_map[image_dict["id"]]
        labels = defaultdict(list)
        h, w, _ = img.shape
        for annotation in img_annotations:
            labels["category"].append(annotation["category_id"])
            bbox = annotation["bbox"]
            xbbox, ybbox, wbbox, hbbox = bbox
            labels["bbox"].append(
                xywh_to_yxyx(xbbox / w, ybbox / h, wbbox / w, hbbox / h)
            )
        return img, labels

    @classmethod
    def from_json_filename(cls, coco_json_filename):
        with open(coco_json_filename, "r") as json_file:
            coco_dict = json.load(json_file)

        return cls(coco_dict)


def xywh_to_yxyx(x, y, w, h):
    return (y, x, y + h, x + w)


def collect_annotations(coco_dict):
    image2annotations_map = defaultdict(list)
    for annotation_dict in coco_dict["annotations"]:
        image2annotations_map[annotation_dict["image_id"]].append(annotation_dict)
    return image2annotations_map

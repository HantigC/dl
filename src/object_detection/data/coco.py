from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path

import cv2
from torch.utils.data.dataset import Dataset


class CocoDetectionDataset(Dataset):
    def __init__(self, coco_dict, images_path, transform=None, target_transform=None):
        self.coco_dict = coco_dict
        self.image2annotation_map = collect_annotations(coco_dict)
        self._id2ordered_id = {
            category_dict["id"]: num
            for num, category_dict in enumerate(coco_dict["categories"])
        }
        self.name2id_map = {
            category_dict["name"]: num
            for num, category_dict in enumerate(coco_dict["categories"])
        }
        self.id2name_map = {v: k for k, v in self.name2id_map.items()}
        self.images_path = Path(images_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.coco_dict["images"])

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        image_dict = self.coco_dict["images"][idx]
        img = cv2.cvtColor(
            cv2.imread(str(self.images_path / image_dict["file_name"])),
            cv2.COLOR_BGR2RGB,
        )
        img_annotations = self.image2annotation_map[image_dict["id"]]
        target = defaultdict(list)
        h, w, _ = img.shape
        for annotation in img_annotations:
            target["labels"].append(self._id2ordered_id[annotation["category_id"]])
            bbox = annotation["bbox"]
            xbbox, ybbox, wbbox, hbbox = bbox
            normalized_bbox_yxhw = (ybbox / h, xbbox / w, hbbox / h, wbbox / w)
            target["boxes"].append(normalized_bbox_yxhw)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

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


def make_id2category_map(coco_dict):
    return {
        category_dict["id"]: category_dict["name"]
        for category_dict in coco_dict["categories"]
    }


def make_category2id_map(coco_dict):
    return {
        category_dict["name"]: category_dict["id"]
        for category_dict in coco_dict["categories"]
    }

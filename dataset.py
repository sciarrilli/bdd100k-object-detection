import os
import json
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import pil_loader


def load_json(f):
    with open(f, 'r') as fp:
        return json.load(fp)


class BDDDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.samples = None
        self.prepare()

    def prepare(self):
        self.samples = []

        label_paths = glob.glob(
            os.path.join(self.root, 'labels/train/*.json'))[:10000]
        image_dir = os.path.join(self.root, 'images/100k/train')

        for label_path in label_paths:
            image_path = os.path.join(
                image_dir,
                os.path.basename(label_path).replace('.json', '.jpg'))
            if os.path.exists(image_path):
                self.samples.append((image_path, label_path))

    def __getitem__(self, index):
        # TODO: handle label dict
        image_path, label_path = self.samples[index]
        #image = pil_loader(image_path)
        img = Image.open(image_path).convert("RGB")
        json_labels = load_json(label_path)

        mapping = {
            'pedestrian': 0,
            'rider': 1,
            'car': 2,
            'truck': 3,
            'bus': 4,
            'train': 5,
            'motorcycle': 6,
            'bicycle': 7,
            'traffic light': 8,
            'traffic sign':9
        }
        
        labels = []
        boxes = []
        for label in json_labels['labels']:
            if label['category'] in mapping:
                #print(label)
                x1 = label['box2d']['x1']
                #print(f'x1 {type(x1)}')
                x2 = label['box2d']['x2']
                y1 = label['box2d']['y1']
                y2 = label['box2d']['y2']
                boxes.append([x1, y1, x2, y2])
                labels.append(mapping[label['category']])
                
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        num_objs = len(boxes)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.LongTensor(labels)
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.samples)
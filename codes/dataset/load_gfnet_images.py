import os
import re
import sys

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CustomDataSet(Dataset):

    def __init__(self, input_dir, image_size, crop_pct, dataset_interpolation):
        self.input_dir = input_dir
        self.input_size = [image_size, image_size]
        self.image_list = sorted(os.listdir(input_dir))
        self.transform = transforms.Compose([
                transforms.Resize(int(image_size/crop_pct), interpolation =dataset_interpolation),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.class_to_idx = np.load(
            'class_to_index.npy', allow_pickle=True).item()

    def __getitem__(self, item):
        img_path = self.image_list[item]
        img = Image.open(os.path.join(self.input_dir, img_path)).convert('RGB')
        img = self.transform(img).float()
        class_str = img_path.split('_')[0]
        return img, self.class_to_idx[class_str], img_path

    def __len__(self):
        return len(self.image_list)


def load_images(input_dir, image_size, crop_pct, dataset_interpolation, batch_size=1):
    """Read png images from input directory in batches.
        Args:
            input_dir: input directory
            batch_size: size of minibatch
            input_height: the array size of input
            input_width: the array size of input
        Return:
            dataloader
    """
    img_set = CustomDataSet(
        input_dir=input_dir,
        image_size=image_size,
        crop_pct=crop_pct,
        dataset_interpolation=dataset_interpolation
        )
    img_loader = DataLoader(img_set, batch_size=batch_size, pin_memory=True)
    return img_loader, img_set.image_list

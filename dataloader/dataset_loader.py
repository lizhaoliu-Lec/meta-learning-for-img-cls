""" Dataloader for all datasets. """

import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

__all__ = ['DatasetLoader']


class DatasetLoader(Dataset):
    """The class to load the dataset"""

    def __init__(self, set_name, dataset_dir, train_aug=False):
        # Set the path according to train, val and test
        allow_set_name = ['train', 'test', 'val']
        if set_name not in allow_set_name:
            raise ValueError('expect set name in `%s`, but got `%s`' % (allow_set_name, set_name))

        THE_PATH = osp.join(dataset_dir, set_name)
        label_list = os.listdir(THE_PATH)

        # Generate empty list for images and labels
        images = []
        labels = []

        # Get folders' name
        folders = [osp.join(THE_PATH, label) for label in label_list if os.path.isdir(osp.join(THE_PATH, label))]

        # TODO, I think there maybe some bug here, because the
        # order of folder may not be exactly every time
        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                images.append(osp.join(this_folder, image_path))
                labels.append(idx)

        # Set images, labels and class number to be assessable from outside
        self.images = images
        self.labels = labels
        self.num_class = len(set(labels))

        # Transformation, sequence matter!
        if train_aug:
            image_size = 80
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.RandomResizedCrop(88),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            image_size = 80
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        path, label = self.images[i], self.labels[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

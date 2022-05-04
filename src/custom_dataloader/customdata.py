import torch
import os
from torch.utils.data import Dataset
import cv2


class CustomDataset(Dataset):
    def __init__(self, folder=None):
        self.root_path = os.path.join('.', 'data', 'Covid19-dataset', str(folder))
        self.classes = os.listdir(self.root_path)
        self.data = []
        self.label_map = {'Covid': 0, 'Normal': 1, 'Viral Pneumonia': 2}
        self.img_dim = (256, 256)
        for class_name in self.classes:
            class_path = os.path.join(self.root_path, class_name)
            for img in os.listdir(class_path):
                img_path = os.path.join(class_path, img)
                self.data.append([img_path, class_name])

    def image_transformation(self,img):
        img = cv2.imread(img)
        img = cv2.resize(img, self.img_dim)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = torch.from_numpy(img)
        img = img / 255
        img.requires_grad = True
        img = img.reshape(256, 256, 1)
        img = img.permute(2, 0, 1)
        return img

    def label_transformations(self,class_id):
        class_id = self.label_map[class_id]
        class_id = torch.tensor(class_id)

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        img, class_id = self.data[idx]
        # Image manipulation
        img = self.image_transformation(img)
        class_id = label_transformations(class_id)

        return img, class_id
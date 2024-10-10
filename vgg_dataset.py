import os
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self, root_dir, classes, channels, transform=None, mode="train"
    ):
        """
        root_dir: 데이터셋의 루트 경로
        classes: 클래스 이름 리스트
        channels: 각 클래스의 채널 폴더 이름 리스트
        transform: 적용할 전처리 작업
        mode: 'train', 'valid', 'test' 중 하나
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
        self.channels = channels
        self.mode = mode

        self.data = {"image": [], "label": []}

        for class_idx, class_name in enumerate(self.classes):
            path_class = os.path.join(self.root_dir, mode, class_name)
            channel_image_paths = []

            for ch in self.channels:
                path_class_ch = os.path.join(path_class, ch)

                channel_images = sorted(os.listdir(path_class_ch))
                channel_image_paths.append(channel_images)

            num_images = len(channel_image_paths[0])
            if not all(
                len(images) == num_images for images in channel_image_paths
            ):
                print(f"채널별 이미지 수가 다릅니다: {class_name}")
                continue

            for i in range(num_images):
                channel_images = []
                for ch_idx, ch in enumerate(self.channels):
                    img_path = os.path.join(
                        path_class, ch, channel_image_paths[ch_idx][i]
                    )
                    if os.path.exists(img_path):
                        channel_images.append(img_path)
                    else:
                        print(f"파일이 존재하지 않습니다: {img_path}")
                        continue

                if len(channel_images) == len(self.channels):
                    label = self.get_label(class_name)
                    self.data["image"].append(channel_images)
                    self.data["label"].append(label)

    def get_label(self, class_name):
        label = [0.0 for _ in self.classes]
        idx = self.classes.index(class_name)
        label[idx] = 1.0
        return label

    def __getitem__(self, index):
        path_imgs = self.data["image"][index]
        label = self.data["label"][index]

        imgs = [np.array(Image.open(path).convert("L")) for path in path_imgs]

        imgs = np.stack(imgs, axis=0)
        imgs = torch.tensor(imgs, dtype=torch.float32)

        if self.transform:
            if isinstance(self.transform, transforms.Compose):
                for t in self.transform.transforms:
                    if not isinstance(t, transforms.ToTensor):
                        imgs = t(imgs)

        label = torch.tensor(label, dtype=torch.float32)
        return imgs, label

    def __len__(self):
        return len(self.data["image"])

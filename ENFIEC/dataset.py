import ENFIEC.config as config
import os
from torch.utils.data import  Dataset
from PIL import Image
from ENFIEC.preprocess_images import resize_maintain_aspect

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Cancer_Dataset_1(Dataset):
    def __init__(self, malignant_3, malignant_4a, benign_3, train=True, transform1=None, transform2=None):
        super(Cancer_Dataset_1, self).__init__()
        self.malignant_3 = malignant_3
        self.malignant_4a = malignant_4a
        self.benign_3 = benign_3
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train

    def __len__(self):
        # 返回恶性数据的长度
        return len(self.benign_3) // 2

    def __getitem__(self, index):
        malignant_3, malignant_3_label = self.malignant_3[index].split(";")
        malignant_4a, malignant_4a_label = self.malignant_4a[index].split(";")

        # 取两倍的良性数据
        benign_3_1, benign_3_label_1 = self.benign_3[index * 2].split(";")
        benign_3_2, benign_3_label_2 = self.benign_3[index * 2 + 1].split(";")

        malignant_3_image = Image.open(malignant_3).convert("RGB")
        malignant_4a_image = Image.open(malignant_4a).convert("RGB")
        benign_3_image_1 = Image.open(benign_3_1).convert("RGB")
        benign_3_image_2 = Image.open(benign_3_2).convert("RGB")

        if self.transform1:
            malignant_3_image = self.transform1(malignant_3_image)
            malignant_4a_image = self.transform2(malignant_4a_image)
            benign_3_image_1 = self.transform2(benign_3_image_1)
            benign_3_image_2 = self.transform2(benign_3_image_2)

        return malignant_3_image, malignant_4a_image, benign_3_image_1, benign_3_image_2


class Cancer_Dataset_2(Dataset):
    def __init__(self, total_data, train=True, transform=None, maintain_ratio=False):
        super(Cancer_Dataset_2, self).__init__()
        self.total_data = total_data
        self.transform = transform
        self.train = train
        self.maintain_ratio = maintain_ratio

    def __len__(self):
        return len(self.total_data)

    def __getitem__(self, index):
        path, label = self.total_data[index].split(";")
        label = label.replace("\n", "")
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path


class Cancer_Dataset(Dataset):
    def __init__(self, annotation_lines, train=True, transform=None, maintain_ratio=False):
        super(Cancer_Dataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.transform = transform
        self.train = train
        self.maintain_ratio = maintain_ratio

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        lesion = self.annotation_lines[index].split(';')[0]  # 获得数据的路径
        image = Image.open(lesion).convert("RGB")

        if self.maintain_ratio:
            image = resize_maintain_aspect(image, desired_size=config.IMAGE_SIZE)
        label = int(self.annotation_lines[index].split(';')[-1].split("\n")[0])
        if self.transform:
            image = self.transform(image)
        return image, label, lesion


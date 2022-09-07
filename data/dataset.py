import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class RetinaFaceDataset(Dataset):
    def __init__(self, txt_path, image_size):
        self.txt_path = txt_path
        self.image_size = image_size
        self.imgs_path, self.words = self.get_labels()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(0.5),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        # 打开图像
        image = cv2.imread(self.imgs_path[index])
        h, w, c = image.shape
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        ws, hs = w / self.image_size[1], h / self.image_size[0]
        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return image, annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))

            # bbox真实框的位置 x1,y1,x2,y2  lx,ly...
            annotation[0, 0] = label[0] / ws
            annotation[0, 1] = label[1] / hs
            annotation[0, 2] = (label[0] + label[2]) / ws
            annotation[0, 3] = (label[1] + label[3]) / hs

            # 人脸关键点位置
            annotation[0, 4] = label[4] / ws
            annotation[0, 5] = label[5] / hs
            annotation[0, 6] = label[7] / ws
            annotation[0, 7] = label[8] / hs
            annotation[0, 8] = label[10] / ws
            annotation[0, 9] = label[11] / hs
            annotation[0, 10] = label[13] / ws
            annotation[0, 11] = label[14] / hs
            annotation[0, 12] = label[16] / ws
            annotation[0, 13] = label[17] / hs
            if annotation[0, 4] < 0:
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)

        '''
        
        可添加自定义数据增强
        
        '''
        target[:, 0:14:2] = target[:, 0:14:2] / self.image_size[1]
        target[:, 1:14:2] = target[:, 1:14:2] / self.image_size[0]

        return self.transform(image), torch.Tensor(target)

    def get_labels(self):
        imgs_path, words = [], []
        f = open(self.txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = self.txt_path.replace('label.txt', 'images/') + path.strip()
                imgs_path.append(path)
            else:
                line = line.strip().split(' ')
                label = [float(i) for i in line]
                labels.append(label)

        words.append(labels)
        return imgs_path, words


if __name__ == '__main__':
    data = RetinaFaceDataset(r'G:\widerFace\widerface\train\label.txt', (640, 640))
    image, label = data[0]
    img = np.array(image.permute(1, 2, 0)).astype(np.float).copy()
    print(image.shape, label)
    label = torch.tensor(label, dtype=torch.int)[:, :-1].reshape(-1, 2).tolist()

    cv2.rectangle(img, tuple(label[0]), tuple(label[1]), (0, 0, 255), 2)
    for i in label[2:]:
        cv2.circle(img, (i[0], i[1]), 2, (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)

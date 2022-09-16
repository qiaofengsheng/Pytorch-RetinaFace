import torch
import tqdm
from torch import nn, optim
from data.config import cfg_mobilenet
from data.dataset import *
from model.retinaface import RetinaFace
from tools.loss import *
from tools.anchor import *

'''
    ToDo
'''


class Train:
    def __init__(self):
        self.net = RetinaFace(cfg_mobilenet)
        self.optimizer = optim.Adam(self.net.parameters(), weight_decay=0.00005)
        self.anchors = Anchor(cfg_mobilenet, (640, 640))
        self.train_dataset = RetinaFaceDataset("/data/face_det/data/widerface/train/label.txt", (640, 640))
        self.train_loader = DataLoader(self.train_dataset, batch_size=cfg_mobilenet['batch_size'], shuffle=True,
                                       num_workers=8,collate_fn=detection_collate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    def train(self):
        params = {
            'threshold':0.35,
            'variances':[0.1,0.2],
        }
        for epoch in range(cfg_mobilenet['epoch']):
            with tqdm.tqdm(self.train_loader) as t1:
                for i, (images, targets) in enumerate(tqdm.tqdm(self.train_loader)):
                    images, targets = images.to(self.device), targets.to(self.device)
                    print(images.shape,targets.shape)
                    predicts = self.net(images)
                    print(predicts[0].shape)
                    print(predicts[1].shape)
                    print(predicts[2].shape)
                    loss_conf, loss_bbox, loss_landmark=retinaface_loss(params,predicts,targets,self.anchors)
                    loss = 2*loss_conf+loss_bbox+loss_landmark
                    t1.set_description('Epoch %i' % epoch)
                    t1.set_postfix(loss=loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            torch.save(self.net.state_dict(),'checkpoints/retinaface.pth')
            print('save model successfully!')

if __name__ == '__main__':
    Train().train()
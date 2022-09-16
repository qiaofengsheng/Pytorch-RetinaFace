import torch
from model.layer_utils import *
import torchvision.models._utils as _utils
from data.config import *


class RetinaFace(nn.Module):
    def __init__(self, config, is_train=True):
        super(RetinaFace, self).__init__()
        self.config = config
        self.is_train = is_train
        backbone = MobileNetV1(config['model_size'])

        self.body = _utils.IntermediateLayerGetter(backbone, config['stage_layers'])
        stage_input_channels = config['input_channels']
        stage_input_channel_list = [
            stage_input_channels * 2,
            stage_input_channels * 4,
            stage_input_channels * 8,
        ]
        output_channels = config['output_channels']
        self.fpn = FPN(stage_input_channel_list, output_channels)
        self.ssh1 = SSH(output_channels, output_channels)
        self.ssh2 = SSH(output_channels, output_channels)
        self.ssh3 = SSH(output_channels, output_channels)

        self.class_head1 = ClassHead(output_channels, 2)
        self.class_head2 = ClassHead(output_channels, 2)
        self.class_head3 = ClassHead(output_channels, 2)

        self.bbox_head1 = BboxHead(output_channels, 2)
        self.bbox_head2 = BboxHead(output_channels, 2)
        self.bbox_head3 = BboxHead(output_channels, 2)

        self.landmarks_head1 = LandmarksHead(output_channels, 2)
        self.landmarks_head2 = LandmarksHead(output_channels, 2)
        self.landmarks_head3 = LandmarksHead(output_channels, 2)

    def forward(self, x):
        stage1 = self.body['stage1'](x)
        stage2 = self.body['stage2'](stage1)
        stage3 = self.body['stage3'](stage2)
        fpn_stages = self.fpn([stage1, stage2, stage3])

        # SSH
        feature1 = self.ssh1(fpn_stages[0])
        feature2 = self.ssh1(fpn_stages[1])
        feature3 = self.ssh1(fpn_stages[2])

        # Head
        class_head = torch.cat([self.class_head1(feature1), self.class_head2(feature2), self.class_head3(feature3)],
                               dim=1)
        bbox_head = torch.cat([self.bbox_head1(feature1), self.bbox_head2(feature2), self.bbox_head3(feature3)], dim=1)
        landmark_head = torch.cat(
            [self.landmarks_head1(feature1), self.landmarks_head2(feature2), self.landmarks_head3(feature3)], dim=1)

        if self.is_train == 'train':
            output = (bbox_head, class_head, landmark_head)
        else:
            output = (bbox_head, torch.softmax(class_head, dim=1), landmark_head)

        return output


if __name__ == '__main__':
    x = torch.randn(1, 3, 640, 640)
    net = RetinaFace(cfg_mobilenet)
    print(net(x)[0].shape)
    print(net(x)[1].shape)
    print(net(x)[2].shape)

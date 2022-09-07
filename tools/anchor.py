from itertools import product
from math import ceil

import torch

import data.config


class Anchor:
    def __init__(self, config, image_size=None):
        self.min_sizes = config['min_sizes']
        self.steps = config['steps']
        self.clip = config['clip']

        self.image_size = image_size  # H W

        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        for k, feature_map in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(0, feature_map[0]), range(0, feature_map[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x / feature_map[1] for x in [j + 0.5]]
                    dense_cy = [y / feature_map[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        output = torch.Tensor(anchors).reshape(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


if __name__ == '__main__':
    anchors = Anchor(data.config.cfg_mobilenet, (640, 640)).get_anchors()
    print(anchors.shape)

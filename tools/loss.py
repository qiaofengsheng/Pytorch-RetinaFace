import torch


def anchor_exchange_xyxy(anchor):
    '''
    将生成的anchor坐标信息由xywh转为xyxy格式
    :param anchor: anchor  16800 x 4  shape: x y w h
    :return: anchor 16800 x 4  shape x1 y1 x2 y2
    '''
    tmp_anchor = anchor
    tmp_anchor[:, 0] = anchor[:, 0] - anchor[:, 2] / 2
    tmp_anchor[:, 1] = anchor[:, 1] - anchor[:, 3] / 2
    tmp_anchor[:, 2] = anchor[:, 0] + anchor[:, 2] / 2
    tmp_anchor[:, 3] = anchor[:, 1] + anchor[:, 3] / 2
    return tmp_anchor


def interest_areas(box1, box2):
    '''
    求真实框与建议框的交集面积
    :param box1:truth bbox :x1,y1,x2,y2   shape: N x 4
    :param box2: anchor bbox :x1,y1,x2,y2   shape: 16800 x 4
    :return: interest areas   shape: N X 16800
    '''
    A = box1.size(0)
    B = box2.size(0)
    box1_1 = box1[:, 2:].unsqueeze(1).expand(A, B, 2)
    box2_1 = box2[:, 2:].unsqueeze(0).expand(A, B, 2)
    box1_2 = box1[:, :2].unsqueeze(1).expand(A, B, 2)
    box2_2 = box2[:, :2].unsqueeze(0).expand(A, B, 2)
    min_xy = torch.min(box1_2, box2_2)
    max_xy = torch.max(box1_1, box2_1)
    w_h = torch.clamp_((max_xy - min_xy), min=0)
    areas = w_h[:, :, 0] * w_h[:, :, 1]
    return areas


def truth_anchor_iou(box1, box2):
    '''
    真实框和建议框的iou计算
    :param box1:truth bbox :x1,y1,x2,y2   shape: N x 4
    :param box2: anchor bbox :x1,y1,x2,y2   shape: 16800 x 4
    :return:  ious   shape: N X 16800
    '''
    interest_area = interest_areas(box1, box2)
    box1_area = ((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])).unsqueeze(1).expand_as(interest_area)
    box2_area = ((box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])).unsqueeze(0).expand_as(interest_area)
    tmp_area = box1_area + box2_area - interest_area
    return interest_area / tmp_area


def anchors_select_strategy(box1, box2):
    '''
        ToDo 建议框筛选策略
    :param box1:
    :param box2:
    :return:
    '''
    pass


def retinaface_loss(predict, target, anchors):
    '''
        ToDo
    损失 : cls loss + bbox loss + landmark loss
    :param predict: 网络预测输出
    :param target: 标签目标信息
    :param anchors: 建议框
    :return: cls_loss + bbox_loss + landmark_loss
    '''
    pass


def bbox_encode(targets, anchors, variances):
    '''
    对目标的真实bounding box 和建议框进行编码计算
    :param targerts: 目标真实坐标  shape: N x 4   x1 y1 x2 y2
    :param anchors: 建议框  shape: 16800 x 4    x,y,w,h
    :param variances: 偏移参数  [0.1,0.2]
    :return:
    '''
    # g_cxcy 为中心点的偏移量
    g_cxcy = (targets[:, :2] + targets[:, 2:]) / 2 - anchors[:, :2]
    g_cxcy = g_cxcy / (variances[0] * anchors[:, 2:])

    # g_wh为宽高的缩放比例 ,这里使用log目的和v3中一样为了保证其为正数
    g_wh = (targets[:, 2:] - targets[:, :2]) / anchors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    g_cxcywh = torch.cat([g_cxcy, g_wh], dim=1)
    return g_cxcywh


def landmark_encode(landmarks, anchors, variances):
    '''
    对人脸关键点坐标进行编码计算
    :param landmarks: shape N x 10   x y
    :param anchors: shape 16800 x 4  cx cy w h
    :param variances: 偏移调整参数
    :return:
    '''
    landmarks = landmarks.reshape(-1, 5, 2)  # N x 5 x 2
    anchors_cx = anchors[:, 0].unsqueeze(0).expand(landmarks.size(0), 5).unsqueeze(2)  # N x 5 x 1
    anchors_cy = anchors[:, 1].unsqueeze(0).expand(landmarks.size(0), 5).unsqueeze(2)  # N x 5 x 1
    anchors_w = anchors[:, 2].unsqueeze(0).expand(landmarks.size(0), 5).unsqueeze(2)  # N x 5 x 1
    anchors_h = anchors[:, 3].unsqueeze(0).expand(landmarks.size(0), 5).unsqueeze(2)  # N x 5 x 1
    anchors = torch.cat([anchors_cx, anchors_cy, anchors_w, anchors_h], dim=2)
    g_cxcy = landmarks[:, :, :2] - anchors[:, :, :2]
    g_cxcy = g_cxcy / (variances[0] * anchors[:, :, 2:])
    g_cxcy = g_cxcy.rehape(-1, 10)
    return g_cxcy


def bbox_decode(targets, anchors, variances):
    '''
    对bounding box进行解码计算，只需要对编码过程进行反算即可
    :param targets: 编码后的bounding box信息
    :param anchors: 建议框
    :param variances: 偏移量调整参数
    :return:
    '''
    cxcy = anchors[:, :2] + targets[:, :2] * variances[0] * anchors[:, 2:]
    cxcy = cxcy * 2
    wh = anchors[:, 2:] * torch.exp(targets[:, 2:] * variances[1])
    wh = (cxcy + wh) / 2
    cxcy = (cxcy - wh) / 2
    cxcywh = torch.cat([cxcy, wh], dim=1)
    return cxcywh


def landmark_decode(targets, anchors, variances):
    '''
    对landmark position进行解码计算，只需要对编码过程进行反算即可
    :param targets: 编码后的landmark position信息
    :param anchors: 建议框
    :param variances: 偏移量调整参数
    :return:
    '''
    targets = torch.cat([
        anchors[:, :2] + targets[:, :2] * variances[0] * anchors[:, 2:],
        anchors[:, :2] + targets[:, 2:4] * variances[0] * anchors[:, 2:],
        anchors[:, :2] + targets[:, 4:6] * variances[0] * anchors[:, 2:],
        anchors[:, :2] + targets[:, 6:8] * variances[0] * anchors[:, 2:],
        anchors[:, :2] + targets[:, 8:10] * variances[0] * anchors[:, 2:],
    ], dim=1)
    return targets


if __name__ == '__main__':
    a = torch.Tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    b = torch.Tensor([[12, 21, 31, 42], [22, 23, 24, 52], [2, 3, 4, 5]])
    # area = interest_areas(a, b)
    c = anchor_exchange_xyxy(b)
    print(c)

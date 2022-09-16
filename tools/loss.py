from base64 import encode
import torch


def xywh_exchange_xyxy(boxes):
    '''
    将生成的anchor坐标信息由xywh转为xyxy格式
    :param anchor: anchor  16800 x 4  shape: x y w h
    :return: anchor 16800 x 4  shape x1 y1 x2 y2
    '''
    tmp_boxes = boxes
    tmp_boxes[:, 0] = tmp_boxes[:, 0] - tmp_boxes[:, 2] / 2
    tmp_boxes[:, 1] = tmp_boxes[:, 1] - tmp_boxes[:, 3] / 2
    tmp_boxes[:, 2] = tmp_boxes[:, 0] + tmp_boxes[:, 2] / 2
    tmp_boxes[:, 3] = tmp_boxes[:, 1] + tmp_boxes[:, 3] / 2
    return tmp_boxes


def xyxy_exchange_xwwh(boxes):
    '''
    将xyxy坐标转为xywh
    '''
    return torch.cat((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2], 1)


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


def anchors_select_strategy(threshold, truths, anchors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    # 计算真实框和anchors的iou
    overlaps = truth_anchor_iou(truths, anchors)  # N x 16800

    # 取每个真实框与anchors中iou最大的anchor
    best_prior_overlaps, best_prior_ids = overlaps.max(1, keep_dim=True)  # N x 1     N x 1
    best_prior_overlaps, best_prior_ids = best_prior_overlaps.squeeze(1), best_prior_ids.squeeze(1)  # N    N
    # 取anchors中与每个真实框iou最大的真实框
    best_truth_overlaps, best_truth_ids = overlaps.max(0, keep_dim=True)  # 1 x 16800   1 x 16800
    best_truth_overlaps, best_truth_ids = best_truth_overlaps.squeeze(0), best_truth_ids.squeeze(0)

    # 保证最大的iou的anchor保留下来
    best_truth_overlaps.index_fill_(0, best_prior_ids, 2)
    # 更新每个anchor对于iou最大的真实框的id
    for i in range(best_prior_ids.size(0)):
        best_truth_ids[best_prior_ids[i]] = i

    # 取出每个anchor与目标框iou最大的真实框
    matches_bbox = truths[best_prior_ids]
    # 取出对应的标签
    conf = labels[best_prior_ids]
    # 小于阈值的标签为背景
    conf[best_prior_ids < threshold] = 0
    # 对bbox，landmarks进行编码，并将结果信息返回
    bboxes = bbox_encode(matches_bbox, anchors, variances)
    matches_landmarks = landms[best_prior_ids]
    landmarks = landmark_encode(matches_landmarks, anchors, variances)
    loc_t[idx] = bboxes  # [num_priors,4]
    conf_t[idx] = conf  # [num_priors] 
    landm_t[idx] = landmarks  # [num_priors,10]


def retinaface_loss(params, predict, target, anchors):
    '''
        ToDo
    损失 : cls loss + bbox loss + landmark loss
    :param predict: 网络预测输出
    :param target: 标签目标信息
    :param anchors: 建议框
    :return: cls_loss + bbox_loss + landmark_loss
    '''
    bbox_data, conf_data, landmark_data = predict
    anchors = anchors
    nums = bbox_data.size(0)
    num_anchors = anchors.size(0)

    bbox_target = torch.Tensor(nums, num_anchors, 4)
    landmark_target = torch.Tensor(nums, num_anchors, 10)
    conf_target = torch.Tensor(nums, num_anchors)
    for idx in range(nums):
        truth_bbox = target[:, :4]
        truth_landmark = target[:, 4:14]
        truth_label = target[:, -1]
        default = anchors.data
        anchors_select_strategy(params['threshold'], truth_bbox, default, params['variances'], truth_label,
                                truth_landmark, bbox_target, conf_target, landmark_target, idx)

    pos1 = conf_target > 0
    num_positive = pos1.long().sum(1, keepdims=True)
    N1 = max(num_positive.data.sum().float, 1)
    pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landmark_data)
    landmark_predict = landmark_data[pos_idx1].reshape(-1, 10)
    landmark_target = landmark_target[pos_idx1].reshape(-1, 10)
    loss_landmark = torch.nn.SmoothL1Loss(reduction='sum').forward(landmark_predict, landmark_target)

    pos2 = conf_target != 0
    conf_target[pos2] = 1
    pos_idx2 = pos1.unsqueeze(pos2.dim()).expand_as(bbox_data)
    bbox_predict = bbox_target[pos_idx2].reshape(-1, 4)
    bbox_target = bbox_target[pos_idx2].reshape(-1, 4)
    loss_bbox = torch.nn.SmoothL1Loss(reduction='sum').forward(bbox_predict, bbox_target)

    loss_conf = torch.nn.CrossEntropyLoss(reduction='sum').forward(conf_data, conf_target.long().reshape(-1))

    return loss_conf, loss_bbox, loss_landmark


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
    c = xywh_exchange_xyxy(b)
    print(c)

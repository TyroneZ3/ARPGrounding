import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision


def IoU(boxA, boxB):
    # order = xyxy
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def isCorrect(bbox_annot, bbox_pred, iou_thr=.5, size_h=224):
    for bbox_p in bbox_pred:
        bbox_p = (np.array(bbox_p) / size_h).tolist()
        for bbox_a in bbox_annot:
            if IoU(bbox_p, bbox_a) >= iou_thr:
                return 1
    return 0


def isCorrectHit(bbox_annot, heatmap, orig_img_shape):
    H, W = orig_img_shape
    heatmap_resized = cv2.resize(heatmap, (W, H))
    max_loc = np.unravel_index(np.argmax(heatmap_resized, axis=None), heatmap_resized.shape)

    # try:
    #     threshold_value = filters.threshold_minimum(heatmap_resized)
    #     labeled_foreground = (heatmap_resized > threshold_value).astype(int)
    #     properties = regionprops(labeled_foreground, heatmap_resized)
    #     center_of_mass = properties[0].centroid
    #     weighted_center_of_mass = properties[0].weighted_centroid
    #     max_loc = weighted_center_of_mass
    # except:
    #     max_loc = np.unravel_index(np.argmax(heatmap_resized, axis=None), heatmap_resized.shape)

    for bbox in bbox_annot:
        if bbox[0] <= max_loc[1] <= bbox[2] and bbox[1] <= max_loc[0] <= bbox[3]:
            return 1
    return 0


def union(bbox):
    if len(bbox) == 0:
        return []
    if type(bbox[0]) == type(0.0) or type(bbox[0]) == type(0):
        bbox = [bbox]
    maxes = np.max(bbox, axis=0)
    mins = np.min(bbox, axis=0)
    return [[mins[0], mins[1], maxes[2], maxes[3]]]


def calc_correctness(annot, heatmap, orig_img_shape):
    # bbox_dict = heat2bbox(heatmap, orig_img_shape)
    size_h = heatmap.shape[-1]
    bbox_dict = generate_bbox(heatmap)
    # bbox, bbox_norm, bbox_score = filter_bbox(bbox_dict=bbox_dict, order='xyxy')
    annot = process_gt_bbox(annot, orig_img_shape)
    bbox_norm_annot = union(annot['bbox_norm'])
    bbox_annot = annot['bbox']
    bbox_dict = union(np.array(bbox_dict)[:, :4])
    bbox_correctness = isCorrect(bbox_norm_annot, bbox_dict, iou_thr=.5, size_h=size_h)
    hit_correctness = isCorrectHit(bbox_annot, heatmap, orig_img_shape)
    # att_correctness = attCorrectness(bbox_annot, heatmap, orig_img_shape)
    # return bbox_correctness, hit_correctness, att_correctness, bbox
    # return bbox_correctness, hit_correctness, att_correctness
    return bbox_correctness, hit_correctness, 0


def process_gt_bbox(annot, orig_img_shape):
    out = {}
    h, w = orig_img_shape
    bbox = torch.tensor(annot).numpy()
    out['bbox'] = bbox.copy()
    bbox[:, 0] = bbox[:, 0] / w
    bbox[:, 1] = bbox[:, 1] / h
    bbox[:, 2] = bbox[:, 2] / w
    bbox[:, 3] = bbox[:, 3] / h
    out['bbox_norm'] = bbox.copy()
    return out


def no_tuple(a):
    out = []
    for item in a:
        out.append(item[0])
    return out


def intensity_to_rgb(intensity, cmap='cubehelix', normalize=False):
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype('float32') * 255.0


def generate_bbox(cam, threshold=0.5, nms_threshold=0.05, max_drop_th=0.5):
    heatmap = intensity_to_rgb(cam, normalize=True).astype('uint8')
    gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)

    thr_val = threshold * np.max(gray_heatmap)

    _, thr_gray_heatmap = cv2.threshold(gray_heatmap,
                                        int(thr_val), 255,
                                        cv2.THRESH_TOZERO)
    try:
        _, contours, _ = cv2.findContours(thr_gray_heatmap,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
    except Exception:
        contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        proposals = [cv2.boundingRect(c) for c in contours]
        # proposals = [(x, y, w, h) for (x, y, w, h) in proposals if h * w > 0.05 * 224 * 224]
        if len(proposals) > 0:
            proposals_with_conf = [thr_gray_heatmap[y:y + h, x:x + w].mean() / 255 for (x, y, w, h) in proposals]
            inx = torchvision.ops.nms(torch.tensor(proposals).float(),
                                      torch.tensor(proposals_with_conf).float(),
                                      nms_threshold)
            estimated_bbox = torch.cat((torch.tensor(proposals).float()[inx],
                                        torch.tensor(proposals_with_conf)[inx].unsqueeze(dim=1)),
                                       dim=1).tolist()
            estimated_bbox = [(x, y, x + w, y + h, conf) for (x, y, w, h, conf) in estimated_bbox
                              if conf > max_drop_th * np.max(proposals_with_conf)]
        else:
            estimated_bbox = [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]]
    else:
        estimated_bbox = [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]]
    return estimated_bbox

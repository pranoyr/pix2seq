import torch


def box_iou_vec(boxes1, boxes2):
    # boxes1: [N,4], boxes2: [M,4]
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)), device=boxes1.device)

    tl = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    br = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)



@torch.no_grad()
def fast_map50(pred_boxes, pred_labels, gt_boxes, gt_labels):
    """
    Calculates a proxy F1-Score @ IoU 0.5.
    """
    if len(pred_boxes) == 0:
        return 0.0
    if len(gt_boxes) == 0:
        return 0.0 # If model predicts something but GT is empty, score is 0

    # 1. Calculate IoU [N_pred, M_gt]
    iou = box_iou_vec(pred_boxes, gt_boxes)
    
    # 2. Mask by Class (Only same-class matches count)
    class_match = pred_labels[:, None] == gt_labels[None, :]
    iou = iou * class_match

    # 3. Find Matches (IoU > 0.5)
    # max(dim=0) checks: "Is this GT box covered by ANY prediction?"
    gt_covered_mask = (iou.max(dim=0).values > 0.5)
    matched_gt_count = gt_covered_mask.sum().item()

    # 4. Calculate Precision & Recall
    # Precision: How many of my predictions were actually useful?
    # (We clamp matched_gt_count to len(pred_boxes) to prevent Precision > 1.0)
    true_positives = min(matched_gt_count, len(pred_boxes))
    
    precision = true_positives / len(pred_boxes)
    recall = matched_gt_count / len(gt_boxes)

    # 5. F1 Score (Harmonic Mean)
    if (precision + recall) == 0:
        return 0.0
        
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
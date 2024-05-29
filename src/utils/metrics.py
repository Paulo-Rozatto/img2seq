import cv2
import numpy as np


def iou(polygon, pred, size=320):

    pred = pred[:, 1:].detach().cpu().numpy() * size
    polygon = polygon[:, 1:].detach().cpu().numpy() * size

    poly = np.array(polygon, dtype=np.int32)
    poly_image = np.zeros((size, size), np.uint8)
    poly_image = cv2.fillPoly(poly_image, [poly], 1)
    poly_image[poly_image > 1] = 1

    pred = np.array(pred, dtype=np.int32)
    pred_image = np.zeros((size, size), np.uint8)
    pred_image = cv2.fillPoly(pred_image, [pred], 1)
    pred_image[pred_image > 1] = 1

    intersection = pred_image * poly_image
    union = (pred_image + poly_image)
    iou = cv2.countNonZero(intersection) / cv2.countNonZero(union)

    return iou

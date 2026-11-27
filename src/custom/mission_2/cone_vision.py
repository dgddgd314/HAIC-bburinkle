"""Detect colored cones in aligned G1 RGB-D images."""
import cv2
import numpy as np


# OpenCV encodes hue on [0, 179].
HUE = {"orange": (3, 24), "green": (36, 89), "blue": (94, 133)}


def cone_points(rgb, depth, k, color):
    """Yield median optical-frame surface points, largest color component first."""
    if rgb.shape[:2] != depth.shape or k[0] <= 0 or k[4] <= 0:
        return
    low, high = HUE[color]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (low, 65, 35), (high, 255, 255))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True):
        left, top, width, height, area = stats[i]
        if area < 16 or width > 2 * height or height > 5 * width:
            continue
        box = np.s_[top:top + height, left:left + width]
        z = depth[box]
        rows, cols = np.nonzero((labels[box] == i) & np.isfinite(z) & (z > 0))
        if len(rows) < max(8, area // 2):
            continue
        distance = z[rows, cols]
        x = (cols + left - k[2]) * distance / k[0]
        y = (rows + top - k[5]) * distance / k[4]
        yield np.median(np.column_stack((x, y, distance)), axis=0)

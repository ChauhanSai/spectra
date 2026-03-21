import cv2
import numpy as np
from config.settings import GRID_SIZE


def normalize01(x):
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def color_contrast_saliency(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_lab = lab.reshape(-1, 3).mean(axis=0)
    dist = np.linalg.norm(lab - mean_lab, axis=2)
    return normalize01(dist)


def edge_saliency(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    return normalize01(mag)


def center_prior(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    yy = (yy - h / 2) / (h / 2)
    xx = (xx - w / 2) / (w / 2)
    d2 = xx * xx + yy * yy
    prior = np.exp(-d2 / (2 * 0.60 * 0.60))
    return prior.astype(np.float32)


def get_salient_cells(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w, _ = img.shape

    sal_color = color_contrast_saliency(img)
    sal_edge = edge_saliency(img)
    sal = 0.75 * sal_color + 0.25 * sal_edge
    sal = cv2.GaussianBlur(sal, (0, 0), 2.5)
    sal = sal * (0.85 + 0.15 * center_prior(h, w))

    cell_h = h // GRID_SIZE
    cell_w = w // GRID_SIZE

    most_scores = {}
    least_scores = {}

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w

            cell_sal = sal[y1:y2, x1:x2].reshape(-1)
            if cell_sal.size == 0:
                most_scores[(row, col)] = 0.0
                least_scores[(row, col)] = 0.0
                continue

            k = max(1, int(0.20 * cell_sal.size))
            topk_mean = float(np.mean(np.partition(cell_sal, -k)[-k:]))
            mean_sal = float(np.mean(cell_sal))

            most_scores[(row, col)] = 0.7 * topk_mean + 0.3 * mean_sal
            least_scores[(row, col)] = mean_sal

    most_salient_cell = max(most_scores, key=most_scores.get)
    least_salient_cell = min(least_scores, key=least_scores.get)

    return most_salient_cell, least_salient_cell

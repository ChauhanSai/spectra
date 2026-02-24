import numpy as np

GRID_SIZE = 4


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _color_contrast_saliency(img_rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    except ImportError:
        lab = img_rgb.astype(np.float32)
    mean_lab = lab.reshape(-1, 3).mean(axis=0)
    dist = np.linalg.norm(lab - mean_lab, axis=2)
    return _normalize01(dist)


def _edge_saliency(img_rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        return _normalize01(mag)
    except ImportError:
        gray = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        gx = np.gradient(gray, axis=1)
        gy = np.gradient(gray, axis=0)
        mag = np.sqrt(gx * gx + gy * gy)
        return _normalize01(mag)


def _center_prior(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy = (yy - h / 2) / max(h / 2, 1)
    xx = (xx - w / 2) / max(w / 2, 1)
    d2 = xx * xx + yy * yy
    return np.exp(-d2 / (2 * 0.60 * 0.60)).astype(np.float32)


def get_salient_cells(img_rgb: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = img_rgb.shape[:2]
    sal_color = _color_contrast_saliency(img_rgb)
    sal_edge = _edge_saliency(img_rgb)
    sal = 0.75 * sal_color + 0.25 * sal_edge
    try:
        import cv2
        sal = cv2.GaussianBlur(sal, (0, 0), 2.5)
    except ImportError:
        pass
    sal = sal * (0.85 + 0.15 * _center_prior(h, w))
    cell_h, cell_w = h // GRID_SIZE, w // GRID_SIZE
    most_scores, least_scores = {}, {}
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cell_sal = sal[y1:y2, x1:x2].reshape(-1)
            if cell_sal.size == 0:
                most_scores[(row, col)] = least_scores[(row, col)] = 0.0
                continue
            k = max(1, int(0.20 * cell_sal.size))
            topk_mean = float(np.mean(np.partition(cell_sal, -k)[-k:]))
            mean_sal = float(np.mean(cell_sal))
            most_scores[(row, col)] = 0.7 * topk_mean + 0.3 * mean_sal
            least_scores[(row, col)] = mean_sal
    most_cell = max(most_scores, key=most_scores.get)
    least_cell = min(least_scores, key=least_scores.get)
    return most_cell, least_cell

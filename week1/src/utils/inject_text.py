import cv2
import numpy as np
from config.settings import GRID_SIZE, LOW_CONTRAST_OFFSET


def clamp_color(x):
    return int(max(0, min(255, x)))


def get_cell_bbox(h, w, row, col):
    cell_h = h // GRID_SIZE
    cell_w = w // GRID_SIZE
    y1 = row * cell_h
    y2 = (row + 1) * cell_h
    x1 = col * cell_w
    x2 = (col + 1) * cell_w
    return x1, y1, x2, y2


def get_avg_bg_color(img_bgr, bbox):
    x1, y1, x2, y2 = bbox
    cell = img_bgr[y1:y2, x1:x2]
    b, g, r = cell.reshape(-1, 3).mean(axis=0)
    return int(b), int(g), int(r)


def pick_text_color(bg_bgr, contrast_level):
    b, g, r = bg_bgr

    if contrast_level == "low_contrast":
        brightness = (b + g + r) / 3
        if brightness > 127:
            return (
                clamp_color(b - LOW_CONTRAST_OFFSET),
                clamp_color(g - LOW_CONTRAST_OFFSET),
                clamp_color(r - LOW_CONTRAST_OFFSET),
            )
        else:
            return (
                clamp_color(b + LOW_CONTRAST_OFFSET),
                clamp_color(g + LOW_CONTRAST_OFFSET),
                clamp_color(r + LOW_CONTRAST_OFFSET),
            )

    color_spread = max(b, g, r) - min(b, g, r)
    if color_spread < 20:
        brightness = (b + g + r) / 3
        return (0, 0, 0) if brightness > 127 else (255, 255, 255)

    if g >= r and g >= b:
        return (0, 0, 255)
    if r >= g and r >= b:
        return (255, 255, 0)
    return (0, 255, 255)


def wrap_lines_for_scale(text, font, scale, thickness, max_w):
    words = text.split()
    if not words:
        return [text]

    lines = []
    cur = words[0]
    for w in words[1:]:
        candidate = f"{cur} {w}"
        (cand_w, _), _ = cv2.getTextSize(candidate, font, scale, thickness)
        if cand_w <= max_w:
            cur = candidate
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def fit_text_block(text, font, thickness, max_w, max_h, size_name):
    if size_name == "large":
        start_scale = 0.95
        min_scale = 0.22
    else:
        start_scale = 0.45
        min_scale = 0.18

    best = None
    scale = start_scale
    while scale >= min_scale:
        lines = wrap_lines_for_scale(text, font, scale, thickness, max_w)
        if len(lines) > 4:
            scale -= 0.03
            continue

        line_sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
        max_line_w = max(w for w, _ in line_sizes)
        line_h = max(h for _, h in line_sizes)
        line_gap = int(line_h * 0.28)
        block_h = len(lines) * line_h + (len(lines) - 1) * line_gap

        if max_line_w <= max_w and block_h <= max_h:
            return lines, scale, line_h, line_gap

        best = (lines, scale, line_h, line_gap)
        scale -= 0.03

    return best


def draw_text_in_cell(img_bgr, text, row, col, size_name, contrast_level):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = get_cell_bbox(h, w, row, col)

    cell_w = x2 - x1
    cell_h = y2 - y1

    bg = get_avg_bg_color(img_bgr, (x1, y1, x2, y2))
    color = pick_text_color(bg, contrast_level)

    thickness = 2 if size_name == "large" else 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    margin_x = int(cell_w * 0.06)
    margin_y = int(cell_h * 0.10)
    max_w = cell_w - 2 * margin_x
    max_h = cell_h - 2 * margin_y

    fit = fit_text_block(text, font, thickness, max_w, max_h, size_name)
    if fit is None:
        return img_bgr
    lines, scale, line_h, line_gap = fit

    block_h = len(lines) * line_h + (len(lines) - 1) * line_gap
    y_cursor = y1 + (cell_h - block_h) // 2 + line_h

    for line in lines:
        (text_w, _), _ = cv2.getTextSize(line, font, scale, thickness)
        x_text = x1 + max(margin_x, (cell_w - text_w) // 2)
        if contrast_level == "high_contrast":
            outline = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
            cv2.putText(
                img_bgr,
                line,
                (x_text, y_cursor),
                font,
                scale,
                outline,
                thickness + 2,
                cv2.LINE_AA,
            )
        cv2.putText(
            img_bgr,
            line,
            (x_text, y_cursor),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y_cursor += line_h + line_gap

    return img_bgr

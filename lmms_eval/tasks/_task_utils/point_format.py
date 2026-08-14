"""Shared parser for Qwen-native point outputs."""

import math
import re

import numpy as np


def parse_point2d(text: str, width: int, height: int) -> np.ndarray:
    """Parse 0-1000 points or unit-normalized decimal points into pixels."""
    pattern = r"[\[\(]\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*[\]\)]"
    points = []
    for xs, ys in re.findall(pattern, text):
        x_value, y_value = float(xs), float(ys)
        uses_unit_grid = ("." in xs or "." in ys) and 0 <= x_value <= 1 and 0 <= y_value <= 1
        if uses_unit_grid:
            x, y = math.floor(x_value * width), math.floor(y_value * height)
        else:
            x, y = math.floor(x_value / 1000 * width), math.floor(y_value / 1000 * height)
        points.append((x, y))
    return np.array(points)

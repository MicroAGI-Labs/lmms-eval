import re
from typing import Any

import numpy as np
from lmms_eval.tasks._task_utils.default_template_yaml import load_default_template_yaml
from lmms_eval.tasks._task_utils.point_format import parse_point2d

PROMPT_SUFFIX_0_999 = "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be integers between 0 and 999, representing the pixel locations scaled to a 1000×1000 grid."
PROMPT_SUFFIX_ORIGINAL = "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
FORMAT = "Return only list of tuples, don't add anything else."
PROMPT_SUFFIX_JSON = "Return points as a JSON list with point_2d coordinates normalized to the range [0, 1000]."


config = load_default_template_yaml(__file__)


def where2place_doc_to_text(doc: dict[str, Any]) -> str:
    if config.get("metadata", {}).get("prompt_suffix_type", {}) == "0_999":
        return f"{doc['question']} {PROMPT_SUFFIX_0_999} {FORMAT}"
    return f"{doc['question']} {PROMPT_SUFFIX_ORIGINAL} {FORMAT}"


def where2place_doc_to_text_json(doc: dict[str, Any]) -> str:
    return f"{doc['question']} {PROMPT_SUFFIX_JSON}"


def where2place_doc_to_visual(doc: dict) -> list:
    return [doc["image"].convert("RGB")]


#  inspired by original repo: https://github.com/wentaoyuan/RoboPoint/blob/master/robopoint/eval/summarize_vqa.py
def _text2pts(
    text: str, width: int = 640, height: int = 480, normalization_constant: int = 1, is_absolute: bool = False
) -> np.ndarray:
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    points = []

    for match in matches:
        vector = [float(num) if "." in num else int(num) for num in match.split(",")]
        if len(vector) == 2:
            x, y = vector
            if not is_absolute:
                x = int(x / normalization_constant * width)
                y = int(y / normalization_constant * height)
            points.append((x, y))

    return np.array(points)


# inspired by original work: https://github.com/wentaoyuan/RoboPoint/blob/master/robopoint/eval/summarize_vqa.py
def _where2place_mask(doc: dict) -> np.ndarray:
    mask = np.array(doc["mask"]) / 255.0
    mask = np.round(mask, 0).astype(int)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def _where2place_accuracy(mask: np.ndarray, points: np.ndarray) -> float:
    acc = 0.0
    if len(points) > 0:
        in_range = (
            (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
        )
        acc = np.concatenate(
            [mask[points[in_range, 1], points[in_range, 0]], np.zeros(points.shape[0] - in_range.sum())]
        ).mean()
    return float(acc)


def _where2place_result(doc: dict, response: str, points: np.ndarray) -> dict[str, dict]:
    acc = _where2place_accuracy(_where2place_mask(doc), points)
    where2place_submission = {
        "id": doc["question_id"],
        "pred": response,
        "parsed_points": list(map(tuple, points)),
        "accuracy": acc,
    }
    return {"where2place_acc": where2place_submission}


def where2place_process_results(doc: dict, result: list[str]) -> dict[str, dict]:
    response = result[0]
    prompt_suffix_type = config.get("metadata", {}).get("prompt_suffix_type", {})
    normalization_constant = 1000 if prompt_suffix_type == "0_999" else 1
    mask = _where2place_mask(doc)
    points = _text2pts(response, mask.shape[1], mask.shape[0], normalization_constant)
    return _where2place_result(doc, response, points)


def where2place_process_results_json(doc: dict, result: list[str]) -> dict[str, dict]:
    response = result[0]
    mask = _where2place_mask(doc)
    points = parse_point2d(response, mask.shape[1], mask.shape[0])
    return _where2place_result(doc, response, points)


def where2place_aggregate_results(results: list[dict]) -> float:
    return float(np.mean([sample["accuracy"] for sample in results]))

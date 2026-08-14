import os
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger as eval_logger

DISCIPLINES = [
    "Tech & Engineering",
    "Science",
    "Health & Medicine",
    "Sports & Arts",
    "Game",
    "Business",
    "Embodied Tasks",
]


replace_prompt = " Please answer yes or no."

# with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         # remove function definition since yaml load cannot handle it
#         if "!function" not in line:
#             safe_data.append(line)

#     config = yaml.safe_load("".join(safe_data))

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# cache_dir = os.path.join(hf_home, cache_dir)
# base_cache_dir = config["dataset_kwargs"]["cache_dir"]
base_cache_dir = os.path.expanduser(hf_home)


VIDEO_SUFFIXES = (".mp4", ".avi")
DATASET_CACHE_REPO = "datasets--MMWorld--MMWorld"
DATASET_REVISION = "34748abf34c1d2fa9470964d5c3156ae8572c8ec"
ANSWER_PATTERN = re.compile(
    r"^\s*(?:(?:the\s+)?(?:(?:best|correct)\s+)?(?:answer|option|choice)"
    r"(?:\s+is\s*:?\s*|\s*:\s*|\s+))?"
    r"\(?([A-D])\)?[.):]?\s*$",
    flags=re.IGNORECASE,
)


def _video_cache_root() -> Path:
    repository_root = Path(base_cache_dir) / "hub" / DATASET_CACHE_REPO
    snapshot_root = repository_root / "snapshots" / DATASET_REVISION
    if not snapshot_root.is_dir():
        raise FileNotFoundError(f"MMWorld Hugging Face snapshot directory does not exist: {snapshot_root}")
    return snapshot_root


@lru_cache(maxsize=1)
def _video_index(cache_root: str) -> dict[str, tuple[Path, ...]]:
    root = Path(cache_root)
    if not root.is_dir():
        raise FileNotFoundError(f"MMWorld video cache directory does not exist: {root}")

    paths_by_name: defaultdict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            paths_by_name[path.name.lower()].append(path)
    return {name: tuple(paths) for name, paths in paths_by_name.items()}


def _resolve_video_path(video_id: str, subdiscipline: str) -> Path:
    cache_root = str(_video_cache_root())
    raw_name = Path(video_id).name
    stem = Path(raw_name).stem if Path(raw_name).suffix.lower() in VIDEO_SUFFIXES else raw_name
    candidate_names = [f"{stem}{suffix}" for suffix in VIDEO_SUFFIXES]
    candidate_names.extend(f"shorts:{stem}{suffix}" for suffix in VIDEO_SUFFIXES)
    matches = {
        path for candidate_name in candidate_names for path in _video_index(cache_root).get(candidate_name.lower(), ())
    }
    if not matches:
        raise FileNotFoundError(f"MMWorld video '{video_id}' was not found under {cache_root}")
    if len(matches) > 1:
        subdiscipline_matches = {
            path for path in matches if subdiscipline.casefold() in {part.casefold() for part in path.parent.parts}
        }
        if len(subdiscipline_matches) == 1:
            return subdiscipline_matches.pop()
        raise ValueError(
            f"MMWorld video '{video_id}' for subdiscipline '{subdiscipline}' is ambiguous: "
            f"{sorted(str(path) for path in matches)}"
        )
    return matches.pop()


def mmworld_doc_to_visual(doc: dict[str, Any]) -> list[str]:
    return [str(_resolve_video_path(doc["video_id"], doc["subdiscipline"]))]


def mmworld_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, str] | None = None) -> str:
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    option_prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = str(doc["options"])
    question = question + "\n" + option
    post_prompt = (
        lmms_eval_specific_kwargs["post_prompt"]
        if "post_prompt" in lmms_eval_specific_kwargs
        else "The best answer is:"
    )
    full_prompt = option_prompt + "\n" + question + "\n" + post_prompt
    return full_prompt


def extract_characters_regex(s: str) -> str:
    match = ANSWER_PATTERN.fullmatch(s)
    return match.group(1).upper() if match is not None else ""


def mmworld_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, str]]:
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    discipline = doc["discipline"]
    data_dict = {
        "video_id": doc["video_id"],
        "discipline": discipline,
        "pred_answer": pred_ans,
        "answer": doc["correct_answer_label"].upper(),
    }

    return {"mmworld_accuracy": data_dict}


def mmworld_aggregate_results(results: list[dict[str, str]]) -> float:
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = {}

    for category in DISCIPLINES:
        key = f"{category}"
        category2score[key] = {"correct": 0, "answered": 0}

    for result in results:
        category = result["discipline"]
        key = f"{category}"
        category2score[key]["answered"] += 1
        category2score[key]["correct"] += result["pred_answer"] == result["answer"]

    for category in DISCIPLINES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if category in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        eval_logger.info(
            f"Evaluation on DISCIPLINES: {category}: {100 * total_correct / total_answered if total_answered > 0 else 0: .1f}%"
        )

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0: .1f}%")
    return 100 * total_correct / total_answered if total_answered > 0 else 0

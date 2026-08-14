import base64
import io
import os
import re
import time
from functools import lru_cache

from loguru import logger as eval_logger
from openai import OpenAI
from PIL import Image

JUDGE_MAX_RETRIES = 6
JUDGE_RETRY_SECONDS = 15

JUDGE_PROMPT = """### Role
You are an expert judge evaluating whether a model answer is correct for a visual question.

Classify the model answer as exactly one of:

1. Correct: It contains the ground truth's core information, has no contradiction, and its granularity is equal to or finer than the ground truth. Irrelevant details are allowed if they do not conflict.
2. Incorrect: It contradicts the ground truth, names the wrong entity/value, or is less specific than the ground truth.
3. Unattempted: It explicitly declines, redirects the user elsewhere, or provides no information from the ground truth without making a contradictory claim.

Ignore differences in formatting, punctuation, language, and abbreviations. A more specific semantically correct answer is Correct. An uncertain answer followed by a wrong specific answer is Incorrect.

Return exactly two lines:
Evaluation: <brief explanation>
Label: <Correct, Incorrect, or Unattempted>

Question: {question}
Model Answer: {model_answer}
Ground Truth Answer: {ground_truth}
"""


@lru_cache(maxsize=1)
def _judge_client() -> OpenAI:
    return OpenAI(api_key=os.environ["JUDGE_API_KEY"], base_url=os.environ["JUDGE_BASE_URL"])


def worldvqa_doc_to_visual(doc: dict) -> list[Image.Image]:
    image = doc["image"]
    if isinstance(image, Image.Image):
        return [image.convert("RGB")]
    if isinstance(image, str):
        if os.path.exists(image):
            return [Image.open(image).convert("RGB")]
        return [Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")]
    if isinstance(image, dict):
        if image["path"] is not None:
            return [Image.open(image["path"]).convert("RGB")]
        return [Image.open(io.BytesIO(image["bytes"])).convert("RGB")]
    raise TypeError(f"Unsupported WorldVQA image type: {type(image).__name__}")


def worldvqa_doc_to_text(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    detail_prompt = (
        "请尽可能提供详细的回答。\n" if doc["language"] == "zh" else "Please provide as much detail as possible.\n"
    )
    return f"{kwargs.get('pre_prompt', '')}{detail_prompt}{doc['question'].strip()}{kwargs.get('post_prompt', '')}"


def _strip_thinking(text: str) -> str:
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    stripped = re.sub(r"<think>.*$", "", stripped, flags=re.DOTALL | re.IGNORECASE).strip()
    if "think>" in stripped.lower():
        stripped = re.split(r"think>", stripped, flags=re.IGNORECASE)[-1].strip()
    return stripped


def _parse_label(judgment: str) -> str:
    labels = re.findall(r"^Label:\s*(Correct|Incorrect|Unattempted)\s*$", judgment, flags=re.MULTILINE | re.IGNORECASE)
    if len(labels) != 1:
        raise ValueError(f"WorldVQA judge returned {len(labels)} parseable labels")
    return labels[0].lower()


def _judge_answer(question: str, prediction: str, ground_truth: str) -> tuple[str, str, str]:
    prompt = JUDGE_PROMPT.format(
        question=question,
        model_answer=_strip_thinking(prediction),
        ground_truth=ground_truth,
    )
    model = os.environ["JUDGE_MODEL_NAME"]
    last_error = "judge did not return a response"
    for attempt in range(JUDGE_MAX_RETRIES):
        try:
            response = _judge_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1024,
            )
            judgment = response.choices[0].message.content or ""
            return _parse_label(judgment), judgment, response.model
        except Exception as error:
            last_error = str(error)
            eval_logger.error(f"WorldVQA judge attempt {attempt + 1} failed: {error}")
            if attempt < JUDGE_MAX_RETRIES - 1:
                time.sleep(JUDGE_RETRY_SECONDS)
    raise RuntimeError(f"WorldVQA judge failed after {JUDGE_MAX_RETRIES} attempts: {last_error}")


def worldvqa_process_results(doc: dict, results: list[str]) -> dict[str, dict]:
    prediction = results[0]
    label, judgment, judge_model = _judge_answer(doc["question"], prediction, doc["answer"])
    return {
        "worldvqa_judge": {
            "score": int(label == "correct"),
            "label": label,
            "judge": judgment,
            "judge_model": judge_model,
            "prediction": prediction,
            "reference": doc["answer"],
        }
    }


def worldvqa_aggregate_judge(results: list[dict]) -> float:
    return sum(result["score"] for result in results) / len(results)

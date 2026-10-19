import os
from functools import lru_cache

from datasets import Dataset
from lmms_eval.tasks.charxiv.constant import REASONING_RESP_INST
from lmms_eval.tasks.charxiv.descriptive_utils import build_descriptive_grading_queries
from lmms_eval.tasks.charxiv.descriptive_utils import descriptive_query_helper
from lmms_eval.tasks.charxiv.descriptive_utils import get_descriptive_result_gpt
from lmms_eval.tasks.charxiv.descriptive_utils import postprocess_descriptive_grading_queries
from lmms_eval.tasks.charxiv.reasoning_utils import build_reasoning_grading_queries
from lmms_eval.tasks.charxiv.reasoning_utils import get_number_instruction
from lmms_eval.tasks.charxiv.reasoning_utils import get_reasoning_result_gpt
from openai import OpenAI
from pydantic import AnyHttpUrl
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class JudgeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: str = Field(min_length=1)
    base_url: AnyHttpUrl
    model: str = Field(min_length=1)


def _judge_config() -> JudgeConfig:
    return JudgeConfig.model_validate(
        {
            "api_key": os.environ["JUDGE_API_KEY"],
            "base_url": os.environ["JUDGE_BASE_URL"],
            "model": os.environ["JUDGE_MODEL_NAME"],
        }
    )


@lru_cache(maxsize=1)
def _get_client(config: JudgeConfig) -> OpenAI:
    """Construct the unpicklable client only after dataset preprocessing."""
    return OpenAI(api_key=config.api_key, base_url=str(config.base_url))


def charxiv_reasoning_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None) -> str:
    inst_category = doc["reasoning_q_source"]
    if inst_category in [1, 2, 3]:
        return REASONING_RESP_INST[inst_category].format(doc["reasoning_q"])
    if inst_category == 4:
        return REASONING_RESP_INST[inst_category].format(doc["reasoning_q"], get_number_instruction(doc["reasoning_a"]))
    raise ValueError(f"Invalid instruction category: {inst_category}")


def charxiv_descriptive_process_docs(dataset: Dataset) -> Dataset:
    """Expand each figure into its four distinct descriptive questions."""
    dataset = dataset.select([index for index in range(len(dataset)) for _ in range(4)])

    def _process_row(example, indice):
        q_number = indice % 4 + 1
        qid = example[f"descriptive_q{q_number}"]
        subplot_loc = example["subplot_loc"]
        if subplot_loc is None:
            subplot_loc = [example["subplot_row"], example["subplot_col"]]
        return {
            **example,
            "descriptive_q": descriptive_query_helper(qid, subplot_loc),
            "descriptive_a": example[f"descriptive_a{q_number}"],
            "qid": qid,
            "question_index": q_number - 1,
        }

    return dataset.map(_process_row, with_indices=True)


def charxiv_descriptive_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None) -> str:
    return doc["descriptive_q"]


def charxiv_doc_to_visual(doc) -> list:
    return [doc["image"].convert("RGB")]


def charxiv_descriptive_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None) -> list[dict]:
    question = charxiv_descriptive_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = charxiv_doc_to_visual(doc)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": visuals[0]},
                {"type": "text", "text": question.strip()},
            ],
        }
    ]


def charxiv_reasoning_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None) -> list[dict]:
    question = charxiv_reasoning_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = charxiv_doc_to_visual(doc)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": visuals[0]},
                {"type": "text", "text": question.strip()},
            ],
        }
    ]


def charxiv_descriptive_process_results(doc, results) -> dict:
    """Judge one question with the native rubric, making its result resumable."""
    config = _judge_config()
    resp_key = f"{doc['figure_path']}_{doc['question_index']}"
    groups = {doc["qid"]: [(resp_key, results[0].strip(), doc["descriptive_a"])]}
    query = build_descriptive_grading_queries(groups, nq_per_query=1)[0]
    judgment = get_descriptive_result_gpt(_get_client(config), query["grading_query"], 1, model=config.model)
    scored = postprocess_descriptive_grading_queries([{**query, **judgment}])[resp_key]
    return {"descriptive_acc": scored}


def charxiv_descriptive_aggregate_results(results) -> float:
    return sum(result["score"] for result in results) / len(results)


def charxiv_reasoning_process_results(doc, results) -> dict:
    """Judge one response before aggregation so completed judgments can be saved."""
    config = _judge_config()
    figure_id = str(doc["figure_path"])
    data = {"inst_category": doc["reasoning_q_source"], "figure_id": figure_id, "answer": doc["reasoning_a"]}
    resp_value = {"raw_question": doc["reasoning_q"], "response": results[0].strip()}
    query = build_reasoning_grading_queries({figure_id: data}, {figure_id: resp_value})[figure_id]
    extracted, score = get_reasoning_result_gpt(_get_client(config), query["grading_query"], model=config.model)
    return {"reasoning_acc": {"resp_id": figure_id, "extracted_answer": extracted, "score": score}}


def charxiv_reasoning_aggregate_results(results) -> float:
    return sum(result["score"] for result in results) / len(results)

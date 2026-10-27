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
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class JudgeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_key: str = Field(min_length=1, repr=False)


def _judge_config() -> JudgeConfig:
    return JudgeConfig(
        model=os.environ["JUDGE_MODEL_NAME"],
        base_url=os.environ["JUDGE_BASE_URL"],
        api_key=os.environ["JUDGE_API_KEY"],
    )


@lru_cache(maxsize=1)
def _get_client(config: JudgeConfig) -> OpenAI:
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def charxiv_reasoning_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    inst_category = doc["reasoning_q_source"]
    if inst_category in [1, 2, 3]:
        question = REASONING_RESP_INST[inst_category].format(doc["reasoning_q"])
    # 4: number-in-general -> need to specify the number of decimal places
    elif inst_category == 4:
        question = REASONING_RESP_INST[inst_category].format(
            doc["reasoning_q"], get_number_instruction(doc["reasoning_a"])
        )
    return question


def charxiv_descriptive_process_docs(dataset: Dataset) -> Dataset:
    figure_count = len(dataset)
    dataset = dataset.repeat(4)

    def _process_row(example, indice):
        question_index = indice // figure_count
        q_number = question_index + 1
        descriptive_q = example[f"descriptive_q{q_number}"]
        qid = descriptive_q
        subplot_loc = example["subplot_loc"]
        if subplot_loc is None:
            subplot_row = example["subplot_row"]
            subplot_col = example["subplot_col"]
            subplot_loc = [subplot_row, subplot_col]
        descriptive_q = descriptive_query_helper(descriptive_q, subplot_loc)
        example["descriptive_q"] = descriptive_q
        example["descriptive_a"] = example[f"descriptive_a{q_number}"]
        return {**example, "qid": qid, "question_index": question_index}

    # Use num_proc=None (single-process, no pickling) to avoid SSLContext
    # serialization errors from the lazy OpenAI client in module globals.
    dataset = dataset.map(_process_row, with_indices=True)
    return dataset


def charxiv_descriptive_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    return doc["descriptive_q"]


def charxiv_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def charxiv_descriptive_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
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


def charxiv_reasoning_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
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


def charxiv_descriptive_process_results(doc, results):
    resp_key = f"{doc['figure_path']}_{doc['question_index']}"
    groups = {doc["qid"]: [(resp_key, results[0].strip(), doc["descriptive_a"])]}
    query = build_descriptive_grading_queries(groups, nq_per_query=1)[0]
    config = _judge_config()
    result = get_descriptive_result_gpt(_get_client(config), query["grading_query"], 1, model=config.model)
    scored = postprocess_descriptive_grading_queries([{**query, **result}])
    return {"descriptive_acc": scored[resp_key]}


def charxiv_descriptive_aggregate_results(results):
    return sum(result["score"] for result in results) / len(results)


def charxiv_reasoning_process_results(doc, results):
    figure_id = doc["figure_path"]
    inst_category = doc["reasoning_q_source"]
    response = results[0].strip()
    answer = doc["reasoning_a"]
    data = {}
    data["inst_category"] = inst_category
    data["figure_id"] = figure_id
    data["answer"] = answer
    resp_value = {"raw_question": doc["reasoning_q"], "response": response}
    query = build_reasoning_grading_queries({figure_id: data}, {str(figure_id): resp_value})[str(figure_id)]
    config = _judge_config()
    extracted, score = get_reasoning_result_gpt(_get_client(config), query["grading_query"], model=config.model)
    return {"reasoning_acc": {"resp_id": figure_id, "extracted_answer": extracted, "score": score}}


def charxiv_reasoning_aggregate_results(results):
    return sum(result["score"] for result in results) / len(results)

import json
import os
from copy import deepcopy

from lmms_eval.tasks.charxiv.constant import REASONING_GRADING_INST
from lmms_eval.tasks.charxiv.constant import REASONING_GRADING_PREFIX
from lmms_eval.tasks.charxiv.constant import REASONING_RESP_INST


def get_reasoning_result_gpt(client, prompt, model, max_retries=10) -> tuple[str, int]:
    """Apply the native rubric without truncating the judge or scoring failures."""
    if max_retries < 1:
        raise ValueError("max_retries must be positive")
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                response_format={"type": "json_object"},
                n=1,
                temperature=0,
                top_p=1,
                seed=42,
            )
            content = json.loads(response.choices[0].message.content)
            ext, scr = content["extracted_answer"], content["score"]
            if scr not in (0, 1):
                raise ValueError(f"Invalid CharXiv reasoning score: {scr}")
            return ext, scr
        except Exception:
            if attempt == max_retries:
                raise
    raise RuntimeError("CharXiv judge retry limit exhausted")


def get_number_instruction(answer) -> str:
    base = answer.split(".")
    whole, decimal = base[0], None if len(base) == 1 else base[1]
    # check if it contains decimal places
    if whole is not None and decimal is None:
        inst = "* Your final answer must be an exact integer."
    elif whole is not None and decimal is not None:
        num_decimal = len(decimal)
        inst = f"* Your final answer must be a number with {num_decimal} decimal places."
    else:
        raise ValueError(f"Invalid answer: {answer}")
    return inst


def build_reasoning_grading_queries(input, resp) -> dict[str, dict[str, str]]:
    queries = {}
    for _, data in input.items():
        figure_id = str(data["figure_id"])
        # question without instruction, response
        query, response = resp[figure_id]["raw_question"], resp[figure_id]["response"]
        # get query for answer type (inst_category), then
        # populate the query with the question, ground truth, and response
        grading_query = REASONING_GRADING_PREFIX + deepcopy(REASONING_GRADING_INST[data["inst_category"]]).replace(
            "<|question|>", query
        ).replace("<|ground_truth|>", data["answer"]).replace("<|response|>", response)
        query = {
            "figure_id": figure_id,
            "grading_query": grading_query,
        }
        queries[figure_id] = query
    return queries


def build_reasoning_queries(data, image_dir) -> dict[str | int, dict[str, str | int]]:
    queries = {}
    for _, d in data.items():
        figure_path = os.path.join(image_dir, f"{d['figure_id']}.jpg")
        inst_category = d["inst_category"]
        # 1: text-in-chart, 2: text-in-general, 3: number-in-chart
        if inst_category in [1, 2, 3]:
            question = REASONING_RESP_INST[inst_category].format(d["query"])
        # 4: number-in-general -> need to specify the number of decimal places
        elif inst_category == 4:
            question = REASONING_RESP_INST[inst_category].format(d["query"], get_number_instruction(d["answer"]))
        else:
            raise ValueError(f"Invalid instruction category: {inst_category}")
        query = {
            "figure_id": d["figure_id"],  # figure_id
            "figure_path": figure_path,  # figure_path
            "inst_category": inst_category,  # instruction category
            "raw_question": d["query"],  # question @@@ without @@@ instruction
            "question": question,  # question with instruction
        }
        queries[d["figure_id"]] = query
    return queries

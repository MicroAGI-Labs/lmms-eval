import json
import os
from copy import deepcopy

from lmms_eval.tasks.charxiv.constant import REASONING_GRADING_INST
from lmms_eval.tasks.charxiv.constant import REASONING_GRADING_PREFIX
from lmms_eval.tasks.charxiv.constant import REASONING_RESP_INST
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class Judgment(BaseModel):
    model_config = ConfigDict(strict=True)

    extracted_answer: str
    score: int = Field(ge=0, le=1)


def get_reasoning_result_gpt(client, prompt, model, max_retries=10):
    curr_retries = 0
    while curr_retries < max_retries:
        try:
            response = (
                client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model,
                    response_format={"type": "json_object"},
                    n=1,
                    temperature=0,
                    top_p=1,
                    seed=42,
                )
                .choices[0]
                .message.content
            )
            judgment = Judgment.model_validate(json.loads(response))
            return judgment.extracted_answer, judgment.score
        except Exception as error:
            curr_retries += 1
            if curr_retries == max_retries:
                raise ValueError("Invalid CharXiv reasoning judgment after all retries") from error
    raise ValueError("Invalid CharXiv retry count: max_retries must be positive")


def get_number_instruction(answer):
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


def build_reasoning_grading_queries(input, resp):
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


def build_reasoning_queries(data, image_dir):
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

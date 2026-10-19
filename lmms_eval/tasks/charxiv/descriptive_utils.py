import json
import os

from lmms_eval.tasks.charxiv.constant import DESCRIPTIVE_GRADING_ICL
from lmms_eval.tasks.charxiv.constant import DESCRIPTIVE_GRADING_PREFIX
from lmms_eval.tasks.charxiv.constant import DESCRIPTIVE_GRADING_QMAP
from lmms_eval.tasks.charxiv.constant import DESCRIPTIVE_RESP_INST


def get_rubric(qid) -> str:
    instruction = None
    if qid in [1]:
        instruction = DESCRIPTIVE_GRADING_ICL["title"]
    if qid in [2, 3, 4, 5, 6, 7]:
        instruction = DESCRIPTIVE_GRADING_ICL["ocr"]
    if qid in [8, 9, 10, 12, 14, 15, 17, 19]:
        instruction = DESCRIPTIVE_GRADING_ICL["quant"]
    if qid in [11]:
        instruction = DESCRIPTIVE_GRADING_ICL["bool"]
    if qid in [13]:
        instruction = DESCRIPTIVE_GRADING_ICL["enum"]
    if qid in [16]:
        instruction = DESCRIPTIVE_GRADING_ICL["trend"]
    if qid in [18]:
        instruction = DESCRIPTIVE_GRADING_ICL["layout"]
    assert instruction is not None, f"Instruction for qid {qid} is not found."
    return instruction


def get_descriptive_result_gpt(client, prompt, length, model, max_retries=10) -> dict:
    """Apply the native batched rubric and propagate exhausted judge failures."""
    if max_retries < 1:
        raise ValueError("max_retries must be positive")
    attempt = 0
    while True:
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
            verify_grading_output(content, length)
            return content
        except Exception:
            attempt += 1
            if attempt >= max_retries:
                raise


def build_json_keys(length) -> str:
    keys = []
    # specify the keys for gpt-4o's json response
    for i in range(1, length + 1):
        keys.append(f"extract_answer_T{i}")
        keys.append(f"score_T{i}")
    return str(keys)


def populate_grading_inputs(batch) -> str:
    query = ""
    for i, (_, response, answer) in enumerate(batch):
        # index, response, answer
        curr_query = f"T{i + 1}:\nResponse {i + 1}: {response}\nGround Truth {i + 1}: {answer}\n\n"
        query += curr_query
    return query


def verify_grading_output(data, length_data) -> bool:
    for i in range(1, length_data + 1):
        if f"extract_answer_T{i}" not in data or f"score_T{i}" not in data:
            raise ValueError(f"Missing CharXiv judgment fields for T{i}: {data}")
        if data[f"score_T{i}"] not in (0, 1):
            raise ValueError(f"Invalid CharXiv descriptive score for T{i}: {data}")
    return True


def preprocess_descriptive_grading_queries(input, resp, num_templates=19) -> dict[int, list[tuple[str, str, str]]]:
    # group the responses based on the template id instead of figure id
    groups = {i: [] for i in range(1, num_templates + 1)}
    for _, data in input.items():
        figure_id = data["figure_id"]
        qids = data["qids"]
        for i, qid in enumerate(qids):
            # figure_id with question index
            resp_key = f"{figure_id}_{i}"
            response = resp[resp_key]["response"]
            answer = data["answers"][i]
            groups[qid].append((resp_key, response, answer))
    return groups


def build_descriptive_grading_queries(groups, nq_per_query=5) -> list[dict[str, str | list[str]]]:
    queries = []
    for qid, data in groups.items():
        # batched evaluation based on number of questions per query (nq_per_query)
        for i in range(0, len(data), nq_per_query):
            # batch: list of tuples (resp_key, response, answer)
            batch = data[i : i + nq_per_query]
            # question based on the template id
            question = DESCRIPTIVE_GRADING_QMAP[qid]
            # build the json keys for GPT-4o's response
            json_keys = build_json_keys(len(batch))
            # populate batch size, question, and json keys spec
            prefix = (
                DESCRIPTIVE_GRADING_PREFIX.replace("<|NUM_TRIPLETS|>", str(len(batch)))
                .replace("<|OVERARCHING_QUESTION|>", question)
                .replace("<|JSON_KEYS|>", json_keys)
            )
            # add in-context grading example based on the template id
            rubric_icl = get_rubric(qid)
            # prompt + example + model responses
            grading_query = prefix + rubric_icl + populate_grading_inputs(batch)
            curr_query = {
                "resp_keys": [d[0] for d in batch],
                "grading_query": grading_query,
            }
            queries.append(curr_query)
    return queries


def postprocess_descriptive_grading_queries(queries) -> dict[str, dict[str, str | int]]:
    scores = {}
    for query in queries:
        # query contains resp_keys, grading_query, extract_answer and score
        resp_keys = query["resp_keys"]
        for i, resp_key in enumerate(resp_keys):
            # extract the answer and score for each response key
            extracted_answer = query[f"extract_answer_T{i + 1}"]
            score = query[f"score_T{i + 1}"]
            # store the extracted answer and score
            scores[resp_key] = {
                "resp_id": resp_key,
                "extracted_answer": extracted_answer,
                "score": score,
            }
    return scores


def descriptive_query_helper(qid, subplot_loc) -> str:
    if qid in [18, 19]:
        # skip subplot location when asking about the layout of the subplots
        return DESCRIPTIVE_RESP_INST[qid]
    if isinstance(subplot_loc, list):
        if subplot_loc[0] == 0:
            # when there is only one subplot
            prefix = "For the current plot, "
        else:
            # when there are multiple subplots
            prefix = f"For the subplot at row {subplot_loc[0]} and column {subplot_loc[1]}, "
    # when subplots do not form a grid
    elif isinstance(subplot_loc, str):
        prefix = f"For {subplot_loc}, "
    else:
        raise ValueError(f"Invalid subplot_loc: {subplot_loc}")
    # return the question with the subplot location
    return DESCRIPTIVE_RESP_INST[qid].format(prefix)


def build_descriptive_quries(data, image_dir) -> dict[str, dict[str, str | int]]:
    queries = {}
    for _, d in data.items():
        figure_path = os.path.join(image_dir, f"{d['figure_id']}.jpg")
        for i in range(len(d["qids"])):
            # mapping from template id and subplot location to the question
            question = descriptive_query_helper(d["qids"][i], d["subplot_loc"])
            curr_query = {
                "figure_id": d["figure_id"],  # figure_id
                "figure_path": figure_path,  # figure_path (dropped later)
                "subq_idx": i,  # index of the (4) questions for the given figure
                "qid": d["qids"][i],  # template id
                "question": question,  # question content
            }
            queries[f"{d['figure_id']}_{i}"] = curr_query
    return queries

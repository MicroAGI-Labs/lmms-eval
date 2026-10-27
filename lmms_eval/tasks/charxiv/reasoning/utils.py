from lmms_eval.tasks._task_utils.reasoning_utils import make_reasoning_doc_to_messages
from lmms_eval.tasks._task_utils.reasoning_utils import make_reasoning_process_results
from lmms_eval.tasks.charxiv.constant import DESCRIPTIVE_RESP_INST
from lmms_eval.tasks.charxiv.constant import REASONING_RESP_INST
from lmms_eval.tasks.charxiv.utils import charxiv_descriptive_process_docs as charxiv_descriptive_process_docs
from PIL import Image

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


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


def charxiv_reasoning_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None) -> str:
    inst_category = doc["reasoning_q_source"]
    if inst_category in [1, 2, 3]:
        question = REASONING_RESP_INST[inst_category].format(doc["reasoning_q"])
    # 4: number-in-general -> need to specify the number of decimal places
    elif inst_category == 4:
        question = REASONING_RESP_INST[inst_category].format(
            doc["reasoning_q"], get_number_instruction(doc["reasoning_a"])
        )
    else:
        raise ValueError(f"Invalid instruction category: {inst_category}")
    return question


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


def charxiv_descriptive_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None) -> str:
    return doc["descriptive_q"]


def charxiv_doc_to_visual(doc) -> list[Image.Image]:
    return [doc["image"].convert("RGB")]


charxiv_descriptive_doc_to_messages_cot = make_reasoning_doc_to_messages(
    charxiv_doc_to_visual, charxiv_descriptive_doc_to_text_cot, system_prompt=SYSTEM_PROMPT
)


charxiv_reasoning_doc_to_messages_cot = make_reasoning_doc_to_messages(
    charxiv_doc_to_visual, charxiv_reasoning_doc_to_text_cot, system_prompt=SYSTEM_PROMPT
)


charxiv_reasoning_process_results = make_reasoning_process_results(
    "charxiv", charxiv_reasoning_doc_to_text_cot, gt_key="reasoning_a"
)


charxiv_descriptive_process_results = make_reasoning_process_results(
    "charxiv", charxiv_descriptive_doc_to_text_cot, gt_key="descriptive_a"
)

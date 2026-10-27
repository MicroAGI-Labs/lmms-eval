import ast
import os
import re
import time
from pathlib import Path

import yaml
from loguru import logger as eval_logger
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
)


def load_phyx_config():
    with open(Path(__file__).parent / "phyx.yaml") as f:
        raw_data = f.readlines()
        safe_data = []
        for line in raw_data:
            if "!function" not in line:
                safe_data.append(line)
        return yaml.safe_load("".join(safe_data))


FAIL_MSG = "Failed to obtain answer via API."
config = load_phyx_config()


def _parse_binary_judgment(response: int | str) -> int | None:
    """Accept a strict binary grade, optionally echoing the native prompt's label."""
    match response:
        case bool():
            return None
        case int() as grade:
            return grade if grade in (0, 1) else None
        case str() as text:
            judgment = re.fullmatch(r"\s*(?:Judegement:\s*)?([01])\s*", text, flags=re.IGNORECASE)
            return int(judgment.group(1)) if judgment is not None else None
        case _:
            return None


class PhyXEvaluator:
    """Stateless native scorer; judge credentials are needed only for API requests."""

    def judger_generate(self, prompt: str, temperature: float = 0, patience: int = 5, sleep_time: float = 0) -> str:
        assert not config["metadata"]["quick_extract"], "To employ LLM to extract answer, please set `quick_extract=False`"
        payload: CompletionCreateParamsNonStreaming = {
            "model": os.environ["JUDGE_MODEL_NAME"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        api_key = os.environ["JUDGE_API_KEY"]
        base_url = os.environ["JUDGE_BASE_URL"]
        while patience > 0:
            patience -= 1
            try:
                with OpenAI(api_key=api_key, base_url=base_url) as client:
                    response = client.chat.completions.create(**payload)
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("PhyX judge returned no text content")
                prediction = content.strip()
                if prediction:
                    return prediction

            except Exception as e:
                if "Rate limit" not in str(e):
                    eval_logger.error(e)

                if "Please reduce the length of the messages" in str(e):
                    eval_logger.error("!!Reduce prompt size")
                    new_size = int(len(prompt) * 0.9)
                    new_start = len(prompt) - new_size
                    prompt = prompt[new_start:]
                    payload["messages"] = [ChatCompletionUserMessageParam(role="user", content=prompt)]

                if sleep_time > 0:
                    time.sleep(sleep_time)
        return FAIL_MSG

    def get_ICE(self):
        example_1 = """
    Ground truth answer: 502 \n
    Predicted answer: The mass of block (B) is:
    [
    \\boxed{ 50 \\sqrt{101} }
    ] \n
    Judegement: 1
    """

        example_2 = """
    Ground truth answer: 46.3 kN \n
    Predicted answer: The tension ( T_B ) in the cable is approximately:
    [
    \\boxed{46300 }
    ] \n
    Judegement: 1
    """

        example_3 = """
    Ground truth answer: 12 m/s \n
    Predicted answer: The speed of the box after 2.00 seconds is:
    [
    \\boxed{11.3, \\text{m/s}}
    ] \n
    Judegement: 0
    """

        example_4 = """
    Ground truth answer: 36.00 kg \n
    Predicted answer: The mass of the hanging block ( m_2 ) must be approximately:
    [
    \\boxed{36.1, \\text\\{kg\\}}
    ] \n
    Judegement: 1
    """

        example_5 = """
    Ground truth answer: 3.2 m \n
    Predicted answer: The stuntman and villain slide approximately \\frac\{10\}{3.1415} meters**.
    Judegement: 1
    """

        return [example_1, example_2, example_3, example_4, example_5]

    def get_ICE_MC(self):
        example_1 = """
    Ground truth answer: A \n
    Predicted answer: A \n
    Judegement: 1
    """

        example_2 = """
    Ground truth answer: B \n
    Predicted answer: A \n
    Judegement: 0
    """

        example_3 = """
    Ground truth answer: C \n
    Predicted answer: ### Step 1: Calculate ( l_1 )
    The lightbulb is ( 2.50, \\text\\{m\\}) above the floor, and the bottom of the mirror is (0.50, \\text\\{m\\}) \
    above the floor. The vertical distance from the lightbulb to the bottom of the mirror is:
    [
    \\Delta y_1 = 2.50, \\text\\{m\\} - 0.50, \\text\\{m\\} = 2.00, \\text\\{m\\}.
    ] \n
    Judegement: 0
    """

        example_4 = """
    Ground truth answer: D \n
    Predicted answer: The correct option is D. \n
    Judegement: 1
    """

        return [example_1, example_2, example_3, example_4]

    def build_phyx_gpt4_prompt(self, line, pred):
        task_description = """
    Please read the following example. Given predicted answer and ground truth answer,
    compare the these two answers, then ONLY output judegement 1/0 for matched/unmatched at the end of the prompt.
    If the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
    If the given predicted mentions "approximately", then allow the Approximation Error, \
    such as 0.49 and approximately 0.5, 0.81 and approximately 0.8. \n
    """
        gt_answer = line["answer"]
        prompt = task_description
        examples = self.get_ICE()
        for example in examples:
            prompt += example + "\n"
        prompt += f"Ground truth answer: {gt_answer} \n"
        prompt += f"Predicted answer: {pred} \n"
        prompt += "Judegement:"
        return prompt

    def build_phyx_gpt4_prompt_MC(self, line, pred):
        task_description = """
    Please read the following example. Given predicted answer and ground truth answer for Multi-Choice question.
    The ground truth answer would be A/B/C/D. The predicted answer would be some words containing A/B/C/D.
    Please compare the these two answers, then ONLY output judegement 1/0 for matched/unmatched at the end of the prompt. \n
    """
        gt_answer = line["answer"]
        prompt = task_description
        examples = self.get_ICE_MC()
        for example in examples:
            prompt += example + "\n"
        prompt += f"Ground truth answer: {gt_answer} \n"
        prompt += f"Predicted answer: {pred} \n"
        prompt += "Judegement:"
        return prompt

    def mapping_str(self, input):
        d = {"\dfrac": "\\frac", "\pi": "3.14"}
        output = input
        for k, v in d.items():
            output = output.replace(k, v)
        return output

    def safe_literal_eval(self, s):
        s = s.strip()
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            pass
        if not s.startswith("{"):
            s = "{" + s
        if not s.endswith("}"):
            s = s + "}"
        s = re.sub(r'([{,]\s*)([^"\{\}\:\,\s]+)\s*:', r'\1"\2":', s)
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return None

    def extract_boxed_content(self, s):
        start = s.find(r"\boxed{")
        if start == -1:
            return None
        content_start = start + len(r"\boxed{")
        rest = s[content_start:]
        depth = 0
        for i, ch in enumerate(rest):
            if ch == "{":
                depth += 1
            elif ch == "}":
                if depth == 0:
                    return rest[:i]
                else:
                    depth -= 1
        return None

    def PhyX_auxeval(self, line):
        log = ""
        retry = 5

        gt_answer = str(line["answer"])
        prediction = line["prediction"]

        tmp = self.PhyX_process_line(line)

        if tmp["extracted"] == "Fail to Call API":
            raise RuntimeError("PhyX model generation failed before open-ended judging")

        if tmp["extracted"] != "SAME as predict":
            prediction = tmp["extracted"]

        if gt_answer.strip().lower() == prediction.strip().lower():
            return dict(log="Matched at string level", res=1, extracted=prediction)

        prompt = self.build_phyx_gpt4_prompt(line, prediction)
        for i in range(retry):
            res = self.judger_generate(prompt, temperature=i * 0.5)
            if FAIL_MSG in res:
                log += f"Try {i}: answer and prediction are {gt_answer} and {prediction}, failed to compare.\n"
            else:
                judgment = _parse_binary_judgment(res)
                if judgment == 1:
                    log += "Compared at semantic level. "
                    log += "Semantic equal via LLM."
                    return dict(log=log, res=1, extracted=prediction)
                if judgment == 0:
                    log += "Compared at semantic level. "
                    log += f"LLM judgement {res}"
                    return dict(log=log, res=0, extracted=prediction)
                log += f"Try {i}: judge returned invalid response {res!r}.\n"
        raise RuntimeError("PhyX open-ended judge failed after all retries")

    def PhyX_auxeval_MC(self, line):
        log = ""
        retry = 5

        gt_answer = str(line["answer"])
        prediction = line["prediction"]

        tmp = self.PhyX_process_line_MC(line)

        if tmp["extracted"] == "Fail to Call API":
            raise RuntimeError("PhyX model generation failed before multiple-choice judging")

        if tmp["extracted"] != "SAME as predict":
            prediction = tmp["extracted"]

        if gt_answer.strip().lower() == prediction.strip().lower():
            return dict(log="Matched at string level", res=1, extracted=prediction)
        else:
            if prediction.strip() in ["A", "B", "C", "D"]:
                return dict(log="Unmatched at string level", res=0, extracted=prediction)

        prompt = self.build_phyx_gpt4_prompt_MC(line, prediction)
        for i in range(retry):
            res = self.judger_generate(prompt, temperature=i * 0.5)
            if FAIL_MSG in res:
                log += f"Try {i}: answer and prediction are {gt_answer} and {prediction}, failed to compare.\n"
            else:
                judgment = _parse_binary_judgment(res)
                if judgment == 1:
                    log += "Compared at semantic level. "
                    log += "Semantic equal via LLM."
                    return dict(log=log, res=1, extracted=prediction)
                if judgment == 0:
                    log += "Compared at semantic level. "
                    log += f"LLM judgement {res}"
                    return dict(log=log, res=0, extracted=prediction)
                log += f"Try {i}: judge returned invalid response {res!r}.\n"
        raise RuntimeError("PhyX multiple-choice judge failed after all retries")

    def PhyX_process_line(self, line):
        ret = {}

        answers = str(line["answer"])

        ret["index"] = line["index"]
        ret["gt"] = answers

        prediction_str = line["prediction"]
        with_reasoning = False
        try:
            pred_dict = self.safe_literal_eval(prediction_str)
            if isinstance(pred_dict, dict) and "content" in pred_dict and pred_dict["content"] != "":
                ret["pred"] = pred_dict["content"].strip()
                with_reasoning = True
        except (ValueError, SyntaxError, TypeError):
            pass

        if not with_reasoning:
            ret["pred"] = prediction_str.strip()

        if ret["pred"] == FAIL_MSG:
            ret["match"] = 0
            ret["extracted"] = "Fail to Call API"
            return ret

        boxed_answer = self.extract_boxed_content(ret["pred"])
        if boxed_answer is not None:
            boxed_answer = self.mapping_str(boxed_answer)
            ret["extracted"] = boxed_answer
        else:
            pattern = r"\b(?:final\s+answer|correct\s+answer)\b[^:：]*[:：]\s*(.*?)(?=\n\n\n|\Z)"
            flags = re.IGNORECASE | re.DOTALL
            match = re.search(pattern, ret["pred"], flags=flags)
            if match:
                extracted_answer = match.group(1)
                extracted_answer = self.mapping_str(extracted_answer)
                ret["extracted"] = extracted_answer
            else:
                ret["extracted"] = "SAME as predict"

        if ret["gt"].strip().lower() == ret["extracted"].strip().lower() or ret["gt"].strip().lower() == ret["pred"].strip().lower() or ret["gt"] in ret["pred"]:
            ret["match"] = 1
            return ret

        ret["match"] = 0
        return ret

    def PhyX_process_line_MC(self, line):
        ret = {}

        answers = str(line["answer"])

        ret["index"] = line["index"]
        ret["gt"] = answers
        ret["pred"] = line["prediction"].strip()

        pattern = r"\b(?:correct|answer|option|Answer|Option|Correct)\b[\s\S]*?([A-D])"
        match = re.search(pattern, ret["pred"])

        if ret["pred"] == FAIL_MSG:
            ret["match"] = 0
            ret["extracted"] = "Fail to Call API"
            return ret

        if match:
            extracted_answer = match.group(1)
            ret["extracted"] = extracted_answer
            if ret["gt"].strip().lower() == extracted_answer.strip().lower():
                ret["match"] = 1
                return ret
        else:
            matches = re.findall(r"([ABCD]):", ret["pred"])
            if matches:
                extracted_answer = matches[-1]
                ret["extracted"] = extracted_answer
                if ret["gt"].strip().lower() == extracted_answer.strip().lower():
                    ret["match"] = 1
                    return ret
            else:
                ret["extracted"] = "SAME as predict"

        if ret["gt"] + ":" in ret["pred"] or ret["gt"] + "**" in ret["pred"] or "**" + ret["gt"] in ret["pred"]:
            ret["match"] = 1
        else:
            ret["match"] = 0

        return ret

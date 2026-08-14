"""NVIDIA Cosmos-Reason2 — thin wrapper over Qwen3-VL with NVIDIA's recommended defaults.

Cosmos-Reason2 (2B / 8B / 32B) is post-trained from Qwen3-VL-Instruct without
architecture changes, so generation goes through ``Qwen3VLForConditionalGeneration``.
The model card at https://huggingface.co/nvidia/Cosmos-Reason2-2B prescribes:

- Sampling decode (``temperature=0.7, top_p=0.8, top_k=20``) — Qwen3-VL-Instruct
  generation_config defaults, inherited unchanged.
- ``max_new_tokens=4096+`` — Cosmos2 emits explicit ``<think>...</think><answer>...</answer>``
  blocks; smaller budgets truncate the thinking phase before the answer appears.
- ``attn_implementation="sdpa"`` and ``fps=4`` from NVIDIA's example inference code.
- ``repetition_penalty=1.05`` — NVIDIA-published Qwen3-VL recipe uses
  ``presence_penalty=1.5`` which is vLLM/sglang-only; ``repetition_penalty`` is the
  HF-transformers analog that prevents the loop pathology observed during evaluation.
- ``max_pixels=786432`` (= 768 vision tokens / frame) — matches qwen-vl-utils'
  internal cap. Without this override, lmms-eval's qwen3_vl class default of
  ``1605632`` (= 1568 tokens / frame) is silently clamped to 786432 every frame
  with a per-frame WARNING log line, since qwen-vl-utils enforces the lower cap
  internally. Setting it explicitly silences the noise and makes the config
  honest about what's actually running.

Task-level ``generation_kwargs`` override these via dict-unpacking order; benchmarks
that hardcode greedy decode (legacy VSI-Bench template) will silently disable the
sampling Cosmos2 was trained for, so prefer to use the patched VSI-Bench template
in this fork's lmms-eval which removes the redundant temperature/do_sample fields.
"""


from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL


@register_model("cosmos_reason2")
class CosmosReason2(Qwen3_VL):
    """Cosmos-Reason2 model — Qwen3-VL backbone with NVIDIA-recommended generation defaults."""

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
        "do_sample": True,
        "num_beams": 1,
    }

    def __init__(
        self,
        pretrained: str = "nvidia/Cosmos-Reason2-2B",
        attn_implementation: str | None = "sdpa",
        fps: int | None = 4,
        max_num_frames: int = 32,
        max_pixels: int = 786432,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            attn_implementation=attn_implementation,
            fps=fps,
            max_num_frames=max_num_frames,
            max_pixels=max_pixels,
            **kwargs,
        )

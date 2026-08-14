"""Gemma 4 (E-series, PLE-based) lmms-eval wrapper.

Mirrors `gemma3.py` and exists because upstream lmms-eval does not yet ship a
class for Google's Gemma 4 release (April 2026 technical report). Maintained
on the `microgemma4` branch of MicroAGI-Labs/lmms-eval, used by the moelite
research project. Intended to be upstreamed after moelite's open-source
release.

Notes on the transformers integration:
- We load via `AutoModelForImageTextToText` so the wrapper stays portable
  across transformers versions that may expose the class as either
  `Gemma3nForConditionalGeneration` (transformers <= 5.x, name retained from
  the E-series 3n line) or a future `Gemma4ForConditionalGeneration`.
- The Gemma 4 / 3n processor does not accept the `max_pixels` / `min_pixels`
  kwargs that the Gemma 3 image processor exposes, so we drop them from
  `AutoProcessor.from_pretrained` and from per-image message dicts.
- Chat template, image-token expansion (image_token_id, soft-token budget)
  and PLE handling all live inside the HF model + processor; the wrapper
  only feeds standardized chat messages through `apply_chat_template`.
"""

import os
import time
import warnings
from types import MethodType
from typing import Callable
from typing import Protocol

import torch
from accelerate import Accelerator
from accelerate import DistributedType
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.media_encoder import encode_image_to_data_url
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers.video_utils import VideoMetadata

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

DEFAULT_MAX_FRAMES = 32


class _VideoProcessor(Protocol):
    num_frames: int | None
    sample_frames: Callable[..., torch.Tensor]


def _validate_batch_size(batch_size: int | str | None) -> int:
    if (type(batch_size) is int and batch_size == 1) or (type(batch_size) is str and batch_size == "1"):
        return 1
    raise ValueError(f"Gemma4 requires batch_size=1 because videos can have different frame counts, got {batch_size!r}")


def _validate_max_num_frames(max_num_frames: int) -> int:
    if type(max_num_frames) is not int or max_num_frames <= 0:
        raise ValueError(f"Gemma4 requires max_num_frames to be a positive integer, got {max_num_frames!r}")
    return max_num_frames


def _sample_frames_up_to_limit(
    video_processor: _VideoProcessor,
    metadata: VideoMetadata,
    num_frames: int | None = None,
    fps: int | float | None = None,
    **_: object,
) -> torch.Tensor:
    if fps is not None:
        raise ValueError("Gemma4 uses a fixed maximum frame count, not FPS sampling")
    requested_frames = num_frames if num_frames is not None else video_processor.num_frames
    if requested_frames is None or requested_frames <= 0:
        raise ValueError(f"Gemma4 requires a positive frame limit, got {requested_frames}")
    if metadata.total_num_frames <= 0:
        raise ValueError(f"Gemma4 requires a non-empty video, got {metadata.total_num_frames} frames")

    sampled_frames = min(requested_frames, metadata.total_num_frames)
    if sampled_frames == 1:
        return torch.zeros(1, dtype=torch.int64)
    return torch.arange(sampled_frames, dtype=torch.int64) * (metadata.total_num_frames - 1) // (sampled_frames - 1)


def _configure_video_processor(video_processor: _VideoProcessor, max_num_frames: int) -> None:
    video_processor.num_frames = _validate_max_num_frames(max_num_frames)
    video_processor.sample_frames = MethodType(_sample_frames_up_to_limit, video_processor)


@register_model("gemma4")
class Gemma4(lmms):
    """
    Gemma 4 Model
    https://huggingface.co/google/gemma-4-E2B-it
    """

    def __init__(
        self,
        pretrained: str = "google/gemma-4-E2B-it",
        device: str | None = "cuda",
        device_map: str | None = "auto",
        batch_size: int | str | None = 1,
        trust_remote_code: bool | None = True,
        use_cache=True,
        attn_implementation: str | None = None,
        max_num_frames: int = DEFAULT_MAX_FRAMES,
        interleave_visuals: bool | None = False,
        system_prompt: str | None = "You are a helpful assistant.",
        reasoning_prompt: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        max_num_frames = _validate_max_num_frames(max_num_frames)
        self.batch_size_per_gpu = _validate_batch_size(batch_size)
        if device is None:
            raise ValueError("device must be set (e.g. 'cuda'), got None from model_args")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # AutoModelForImageTextToText resolves to whichever Gemma 4 class the
        # installed transformers exposes (Gemma3nForConditionalGeneration in
        # current releases; a future Gemma4ForConditionalGeneration would also
        # be picked up automatically).
        self._model = AutoModelForImageTextToText.from_pretrained(pretrained, **model_kwargs).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code, device_map=self.device_map
        )
        self.processor = AutoProcessor.from_pretrained(pretrained)
        _configure_video_processor(self.processor.video_processor, max_num_frames)

        self._config = self._model.config
        # vsibench avg 2758 in-tok/sample (32 video frames × 86 image tokens); 2048 silently
        # truncated late video frames. 4096 covers full vsibench distribution at <1 GB extra.
        self._max_length = kwargs.get("max_length", 4096)
        self._model.tie_weights()
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
        self.model.eval()

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        # return self.tokenizer.eod_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("Not implemented for Gemma4.")

    def flatten(self, input: list[list]) -> list:
        """Flatten a nested list into a single list.

        Args:
            input: A nested list structure

        Returns:
            A flattened single-level list
        """
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _encode_image_data_url(self, image: Image.Image) -> str:
        return encode_image_to_data_url(
            image,
            image_format="JPEG",
            mime_type="image/jpeg",
            convert_rgb=True,
            quality=85,
        )

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate text completions for given requests.

        Args:
            requests: List of Instance objects containing generation requests

        Returns:
            List of generated text responses
        """
        res = []
        request_latencies_seconds = []
        total_generated_tokens = 0

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(
                    f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}"
                )

            # Avoid using '\n\n' as a stopper to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]

                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                for visual in visual_list[i]:
                    try:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            if not os.path.exists(visual):
                                eval_logger.warning(f"Video file not found: {visual}")
                                continue
                            processed_visuals.append({"type": "video", "video": visual})
                        elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                            processed_visuals.append({"type": "image", "image": self._encode_image_data_url(visual)})
                    except Exception as e:
                        eval_logger.error(f"Failed to process visual: {e}")
                        continue

                message.append(
                    {
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}],
                    }
                )

                batched_messages.append(message)

            # Dynamic padding (left-pad to longest in batch) instead of max_length so
            # batch_size > 1 stops wasting prefill on padded text tokens; vsibench
            # video inputs (avg 2758 tokens) drive the batch ceiling, not the
            # 128-token question text. Removes silent truncation when any sample
            # exceeds self._max_length.
            inputs = self.processor.apply_chat_template(
                batched_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                processor_kwargs={
                    "return_tensors": "pt",
                    "padding": True,
                    "pad_to_multiple_of": 8,
                },
            ).to(self.model.device, dtype=torch.bfloat16)

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}

            temperature = current_gen_kwargs["temperature"]
            if temperature is not None and temperature > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            generation_start = time.perf_counter()
            cont = self.model.generate(
                **inputs,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            request_latencies_seconds.append(time.perf_counter() - generation_start)

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            total_generated_tokens += sum(len(token_ids) for token_ids in generated_ids_trimmed)
            answers = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        if request_latencies_seconds:
            total_elapsed_time = sum(request_latencies_seconds)
            log_metrics(
                total_elapsed_time=total_elapsed_time,
                total_gen_tokens=total_generated_tokens,
                avg_speed=total_generated_tokens / total_elapsed_time,
                additional_metrics={
                    "total_requests": len(request_latencies_seconds),
                    "request_latencies_seconds": request_latencies_seconds,
                },
            )

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        """Generate text in a multi-round conversation format.

        Args:
            requests: List of Instance objects for multi-round generation

        Returns:
            List of generated responses

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("TODO: Implement multi-round generation")

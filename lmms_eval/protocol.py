import os
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Union

import numpy as np
from PIL import Image
from pydantic import BaseModel

from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.media_encoder import encode_image_to_base64

# Optional video processing dependencies
VideoReader, _has_decord = optional_import("decord", "VideoReader")
cpu, _ = optional_import("decord", "cpu")
fetch_video, _has_qwen_vl = optional_import("qwen_vl_utils", "fetch_video")
FRAME_FACTOR, _ = optional_import("qwen_vl_utils.vision_process", "FRAME_FACTOR")


def _fetch_openai_video(video: Any, video_kwargs: dict[str, Any]):
    if fetch_video is None:
        raise ImportError(
            "qwen_vl_utils is required for video processing. Please install it with: pip install qwen-vl-utils"
        )

    converted_single_frame = False
    resolved_video_kwargs = video_kwargs
    if _has_decord and VideoReader is not None and cpu is not None and isinstance(video, str) and Path(video).is_file():
        # The probe must not bypass qwen_vl_utils' configured backend handling.
        try:
            video_reader = VideoReader(video, ctx=cpu(0))
            total_frames = len(video_reader)
            single_frame = Image.fromarray(video_reader[0].asnumpy()) if total_frames == 1 else None
        except Exception:
            total_frames = 0
            single_frame = None
        if single_frame is not None:
            video = [single_frame]
            converted_single_frame = True
        elif total_frames > 1 and "nframes" in video_kwargs:
            if FRAME_FACTOR is None:
                raise ImportError("qwen_vl_utils.vision_process.FRAME_FACTOR is required for video processing")
            rounded_nframes = round(video_kwargs["nframes"] / FRAME_FACTOR) * FRAME_FACTOR
            if rounded_nframes > total_frames:
                resolved_video_kwargs = {
                    **video_kwargs,
                    "nframes": total_frames // FRAME_FACTOR * FRAME_FACTOR,
                }

    video_input = fetch_video({"type": "video", "video": video, **resolved_video_kwargs})
    return video_input[:1] if converted_single_frame else video_input


class ChatTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatImageContent(BaseModel):
    type: Literal["image"] = "image"
    url: Any


class ChatVideoContent(BaseModel):
    type: Literal["video"] = "video"
    url: Any


class ChatAudioContent(BaseModel):
    type: Literal["audio"] = "audio"
    url: Any


ChatContent = Union[ChatTextContent, ChatImageContent, ChatVideoContent, ChatAudioContent]


class ChatMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: list[ChatContent]


class ChatMessages(BaseModel):
    messages: list[ChatMessage]

    def extract_media(self):
        images = []
        videos = []
        audios = []

        for message in self.messages:
            for content in message.content:
                if content.type == "image":
                    images.append(content.url)
                elif content.type == "video":
                    videos.append(content.url)
                elif content.type == "audio":
                    audios.append(content.url)

        return images, videos, audios

    def to_hf_messages(self, video_kwargs: dict[str, str] | None = None):
        if video_kwargs is None:
            video_kwargs = {}
        _num_frames = video_kwargs.get("nframes", 32)  # noqa: F841
        hf_messages = []
        for message in self.messages:
            hf_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    hf_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    hf_message["content"].append({"type": "image", "image": content.url})
                elif content.type == "video":
                    hf_message["content"].append({"type": "video", "video": content.url, **video_kwargs})
                elif content.type == "audio":
                    hf_message["content"].append({"type": "audio", "audio": content.url})
            hf_messages.append(hf_message)
        return hf_messages

    def to_openai_messages(self, video_kwargs: dict[str, str] | None = None):
        if video_kwargs is None:
            video_kwargs = {}
        openai_messages = []
        encode_cache: dict[tuple[object, ...], str] = {}
        image_format = os.getenv("LMMS_IMAGE_ENCODE_FORMAT", "PNG").upper()
        mime_type = f"image/{'jpeg' if image_format == 'JPG' else image_format.lower()}"
        quality = int(os.getenv("LMMS_IMAGE_JPEG_QUALITY", "85")) if image_format in {"JPEG", "JPG", "WEBP"} else None
        for message in self.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    openai_message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{self.encode_image(content.url, encode_cache, image_format, quality)}"
                            },
                        }
                    )
                elif content.type == "video":
                    video_input = _fetch_openai_video(content.url, video_kwargs)
                    for frame in video_input:
                        image = Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
                        openai_message["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{self.encode_image(image, encode_cache, image_format, quality)}"
                                },
                            }
                        )
                # TODO, audio hasn't been implemented yet
                elif content.type == "audio":
                    openai_message["content"].append({"type": "audio_url", "audio_url": {"url": content.url}})
            openai_messages.append(openai_message)
        return openai_messages

    def to_qwen3_vl_openai_messages(self, video_kwargs: dict[str, str] | None = None):
        if video_kwargs is None:
            video_kwargs = {}
        openai_messages = []
        encode_cache: dict[tuple[object, ...], str] = {}
        image_format = os.getenv("LMMS_IMAGE_ENCODE_FORMAT", "PNG").upper()
        mime_type = f"image/{'jpeg' if image_format == 'JPG' else image_format.lower()}"
        quality = int(os.getenv("LMMS_IMAGE_JPEG_QUALITY", "85")) if image_format in {"JPEG", "JPG", "WEBP"} else None
        for message in self.messages:
            openai_message = {"role": message.role, "content": []}
            for content in message.content:
                if content.type == "text":
                    openai_message["content"].append({"type": "text", "text": content.text})
                elif content.type == "image":
                    openai_message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{self.encode_image(content.url, encode_cache, image_format, quality)}"
                            },
                        }
                    )
                elif content.type == "video":
                    if fetch_video is None:
                        raise ImportError(
                            "qwen_vl_utils is required for video processing. Please install it with: pip install qwen-vl-utils"
                        )
                    video_input, fps = fetch_video(
                        {"type": "video", "video": content.url, **video_kwargs},
                        return_video_metadata=True,
                        return_video_sample_fps=True,
                    )
                    frames, video_metadata = video_input
                    timestamps = self._calculate_timestamps(video_metadata)
                    for frame, timestamp in zip(frames, timestamps):
                        image = Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
                        openai_message["content"].append({"type": "text", "text": f"<{timestamp:.1f} seconds>"})
                        openai_message["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{self.encode_image(image, encode_cache, image_format, quality)}"
                                },
                            }
                        )
                # TODO, audio hasn't been implemented yet
                elif content.type == "audio":
                    openai_message["content"].append({"type": "audio_url", "audio_url": {"url": content.url}})
            openai_messages.append(openai_message)
        return openai_messages

    def _calculate_timestamps(self, video_metadata: dict[str, Any]):
        indices = video_metadata["frames_indices"]
        if not isinstance(indices, list):
            indices = indices.tolist()
        fps = video_metadata["fps"]
        # Note this is a hardcode value for Qwen3-VL, should only be used for Qwen3-VL
        merge_size = 2
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / fps for idx in indices]
        # timestamps = [(timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)]
        return timestamps

    def encode_image(
        self,
        image: Image.Image | str,
        cache: dict[tuple[object, ...], str] | None = None,
        image_format: str = "PNG",
        quality: int | None = None,
    ):
        normalized_image_format = image_format.upper()
        return encode_image_to_base64(
            image,
            image_format=normalized_image_format,
            convert_rgb=normalized_image_format in {"JPEG", "JPG", "WEBP"},
            quality=quality,
            copy_if_pil=False,
            cache=cache,
            use_path_cache=True,
        )

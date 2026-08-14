from __future__ import annotations

import unittest
from types import SimpleNamespace

from lmms_eval.models import get_model
from lmms_eval.models.chat.openai import OpenAICompatible as ChatOpenAICompatible
from lmms_eval.models.simple.openai import OpenAICompatible as SimpleOpenAICompatible


def _fake_response(
    content: str | None = "ok",
    reasoning: str | None = None,
    reasoning_content: str | None = None,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, reasoning=reasoning, reasoning_content=reasoning_content)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    return SimpleNamespace(choices=[choice], usage=None)


class _CaptureCompletions:
    def __init__(self, responses: list[SimpleNamespace] | None = None) -> None:
        self.payloads: list[dict] = []
        self.responses = responses or [_fake_response()]

    def create(self, **payload):
        self.payloads.append(payload)
        return self.responses[min(len(self.payloads) - 1, len(self.responses) - 1)]


def _request(*args) -> SimpleNamespace:
    return SimpleNamespace(args=args)


def _configure_openai_model(model, completions: _CaptureCompletions, *, model_version: str = "gpt-4o") -> None:
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    model.model_version = model_version
    model.max_retries = 1
    model.retry_backoff_s = 0
    model.num_concurrent = 1
    model.adaptive_concurrency = False
    model.adaptive_config = SimpleNamespace(max_concurrency=1)
    model.prefix_aware_queue = False
    model.prefix_hash_chars = 256
    model.thinking_mode = None
    model.max_frames_num = 1
    model.video_fps = None
    model._rank = 0
    model.task_dict = {"demo": {"test": [{"id": 0}]}}


class TestOpenAICompatibleMaxTokens(unittest.TestCase):
    def test_registry_selects_chat_backend(self) -> None:
        self.assertIs(get_model("openai"), ChatOpenAICompatible)

    def test_simple_backend_preserves_requested_max_new_tokens(self) -> None:
        completions = _CaptureCompletions()
        model = SimpleOpenAICompatible.__new__(SimpleOpenAICompatible)
        _configure_openai_model(model, completions)

        model.generate_until(
            [
                _request(
                    "Describe the image",
                    {"max_new_tokens": 8192, "temperature": 0},
                    lambda _doc: None,
                    0,
                    "demo",
                    "test",
                )
            ]
        )

        self.assertEqual(completions.payloads[0]["max_tokens"], 8192)

    def test_simple_backend_forwards_minimax_thinking_mode(self) -> None:
        completions = _CaptureCompletions()
        model = SimpleOpenAICompatible.__new__(SimpleOpenAICompatible)
        _configure_openai_model(model, completions)
        model.thinking_mode = "disabled"

        model.generate_until(
            [
                _request(
                    "Choose the best option",
                    {"max_new_tokens": 8192, "temperature": 0},
                    lambda _doc: None,
                    0,
                    "demo",
                    "test",
                )
            ]
        )

        self.assertEqual(
            completions.payloads[0]["extra_body"],
            {"chat_template_kwargs": {"thinking_mode": "disabled"}},
        )

    def test_simple_backend_retries_empty_endpoint_content(self) -> None:
        completions = _CaptureCompletions([_fake_response(None), _fake_response("A")])
        model = SimpleOpenAICompatible.__new__(SimpleOpenAICompatible)
        _configure_openai_model(model, completions)
        model.max_retries = 2

        responses = model.generate_until(
            [
                _request(
                    "Choose the best option",
                    {"max_new_tokens": 8192, "temperature": 0},
                    lambda _doc: None,
                    0,
                    "demo",
                    "test",
                )
            ]
        )

        self.assertEqual(len(completions.payloads), 2)
        self.assertEqual(responses[0].text, "A")

    def test_chat_backend_preserves_requested_max_new_tokens(self) -> None:
        completions = _CaptureCompletions()
        model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
        _configure_openai_model(model, completions)

        model.generate_until(
            [
                _request(
                    "",
                    lambda _doc: [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Describe this"}],
                        }
                    ],
                    {"max_new_tokens": 32768, "temperature": 0},
                    0,
                    "demo",
                    "test",
                )
            ]
        )

        self.assertEqual(completions.payloads[0]["max_tokens"], 32768)

    def test_chat_backend_forwards_minimax_thinking_mode(self) -> None:
        completions = _CaptureCompletions()
        model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
        _configure_openai_model(model, completions)
        model.thinking_mode = "disabled"

        model.generate_until(
            [
                _request(
                    "",
                    lambda _doc: [{"role": "user", "content": [{"type": "text", "text": "Choose one"}]}],
                    {"max_new_tokens": 8192, "temperature": 0},
                    0,
                    "demo",
                    "test",
                )
            ]
        )

        self.assertEqual(
            completions.payloads[0]["extra_body"],
            {"chat_template_kwargs": {"thinking_mode": "disabled"}},
        )

    def test_chat_backend_retries_empty_endpoint_content(self) -> None:
        completions = _CaptureCompletions([_fake_response(None), _fake_response("A")])
        model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
        _configure_openai_model(model, completions)
        model.max_retries = 2

        responses = model.generate_until(
            [
                _request(
                    "",
                    lambda _doc: [{"role": "user", "content": [{"type": "text", "text": "Choose one"}]}],
                    {"max_new_tokens": 8192, "temperature": 0},
                    0,
                    "demo",
                    "test",
                )
            ]
        )

        self.assertEqual(len(completions.payloads), 2)
        self.assertEqual(responses[0].text, "A")

    def test_chat_backend_uses_reasoning_only_response_when_thinking_is_allowed(self) -> None:
        cases = [
            (None, _fake_response(None, reasoning="A")),
            ("enabled", _fake_response(None, reasoning_content="A")),
            ("adaptive", _fake_response(None, reasoning="A")),
        ]

        for thinking_mode, response in cases:
            with self.subTest(thinking_mode=thinking_mode):
                completions = _CaptureCompletions([response])
                model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
                _configure_openai_model(model, completions)
                model.thinking_mode = thinking_mode

                responses = model.generate_until(
                    [
                        _request(
                            "",
                            lambda _doc: [{"role": "user", "content": [{"type": "text", "text": "Choose one"}]}],
                            {"max_new_tokens": 8192, "temperature": 0},
                            0,
                            "demo",
                            "test",
                        )
                    ]
                )

                self.assertEqual(len(completions.payloads), 1)
                self.assertEqual(responses[0].text, "A")

    def test_chat_backend_ignores_reasoning_only_response_when_thinking_is_disabled(self) -> None:
        completions = _CaptureCompletions([_fake_response(None, reasoning_content="internal"), _fake_response("A")])
        model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
        _configure_openai_model(model, completions)
        model.thinking_mode = "disabled"
        model.max_retries = 2

        responses = model.generate_until(
            [
                _request(
                    "",
                    lambda _doc: [{"role": "user", "content": [{"type": "text", "text": "Choose one"}]}],
                    {"max_new_tokens": 8192, "temperature": 0},
                    0,
                    "demo",
                    "test",
                )
            ]
        )

        self.assertEqual(len(completions.payloads), 2)
        self.assertEqual(responses[0].text, "A")

    def test_chat_reasoning_models_use_requested_completion_tokens(self) -> None:
        completions = _CaptureCompletions()
        model = ChatOpenAICompatible.__new__(ChatOpenAICompatible)
        _configure_openai_model(model, completions, model_version="gpt-5")

        model.generate_until(
            [
                _request(
                    "",
                    lambda _doc: [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Reason carefully"}],
                        }
                    ],
                    {"max_new_tokens": 32768, "temperature": 0.7},
                    0,
                    "demo",
                    "test",
                )
            ]
        )

        self.assertNotIn("max_tokens", completions.payloads[0])
        self.assertEqual(completions.payloads[0]["max_completion_tokens"], 32768)


if __name__ == "__main__":
    unittest.main()

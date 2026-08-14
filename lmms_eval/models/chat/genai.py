"""Vertex AI / Gemini lmms-eval model class.

Targets the Vertex OpenAI-compatible chat-completions endpoint with a
token-refresh httpx auth hook so the SA bearer token (1h default, 12h max
even with the org-policy lifetime extension) mints fresh before expiry on
eval runs that exceed it. Reuses chat/openai's concurrency + retry + budget
machinery — only auth and base URL change.

Auth: ``GOOGLE_APPLICATION_CREDENTIALS`` env var points at the SA JSON.
``google.auth`` marks credentials as ``.expired`` once < ~5 min remain on the
cached token, so the hook re-mints well before the previous token would 401
mid-flight. Vertex's OpenAI-compat endpoint URL differs between regional
locations and the special ``global`` endpoint (e.g. gemini-3.1-flash-lite is
``global``-only) — :func:`_vertex_openai_base_url` handles both.
"""

from __future__ import annotations

import os

import google.auth.transport.requests
import httpx
from google.oauth2 import service_account
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.openai import OpenAICompatible as ChatOpenAICompatible
from openai import OpenAI

_VERTEX_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


def _vertex_openai_base_url(project: str, location: str) -> str:
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"


@register_model("genai")
class GenAIVertex(ChatOpenAICompatible):
    def __init__(
        self,
        model_version: str = "google/gemini-3.1-flash-lite",
        model: str | None = None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
        **kwargs,
    ) -> None:
        project = vertex_project or os.environ.get("VERTEX_PROJECT", "")
        location = vertex_location or os.environ.get("VERTEX_LOCATION", "us-central1")
        if not project:
            raise ValueError(
                "VERTEX_PROJECT not configured. Set it via --model_args vertex_project=... "
                "or the VERTEX_PROJECT env var."
            )

        sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not sa_path:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS not set. The genai model class authenticates via a "
                "service account JSON; for k8s mount the vertex-sa-secret Secret and point this env "
                "var at the file (see k8s/skunkworks/eval_vlm_capability_genai.yaml)."
            )
        credentials = service_account.Credentials.from_service_account_file(sa_path, scopes=list(_VERTEX_SCOPES))

        def _auth_hook(request: httpx.Request) -> None:
            if not credentials.valid:
                credentials.refresh(google.auth.transport.requests.Request())
            request.headers["Authorization"] = f"Bearer {credentials.token}"

        http_client = httpx.Client(event_hooks={"request": [_auth_hook]}, timeout=60.0)

        # Vertex publisher-model id at the OpenAI-compat endpoint takes the
        # ``google/<model>`` form (e.g. ``google/gemini-3.1-flash-lite``); our
        # ``MODEL_PATH`` already matches that shape so ``model_version`` passes
        # through to chat.completions.create without further normalisation.
        base_url = _vertex_openai_base_url(project, location)
        super().__init__(
            model_version=model_version,
            model=model,
            base_url=base_url,
            api_key="vertex-sa-credentials-via-event-hook",
            **kwargs,
        )
        # Swap in the auth-hook-enabled client. The parent's constructor built
        # an OpenAI client with a vanilla httpx client (no auth hook) that we
        # let it construct so all the rest of the parent's wiring still runs.
        self.client = OpenAI(
            api_key="vertex-sa-credentials-via-event-hook",
            base_url=base_url,
            http_client=http_client,
        )

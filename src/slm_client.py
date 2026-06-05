from __future__ import annotations

import json
import time
from dataclasses import dataclass
from itertools import cycle
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_SLM_WORKERS = ["cidt-slm-a", "cidt-slm-b", "cidt-slm-c"]


@dataclass(frozen=True)
class SlmResponse:
    model: str
    content: str
    elapsed_seconds: float


class SlmLoadBalancer:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:1234/v1",
        models: Iterable[str] = DEFAULT_SLM_WORKERS,
        timeout_seconds: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.models = [model for model in models if model]
        if not self.models:
            raise ValueError("At least one SLM model identifier is required.")
        self.timeout_seconds = timeout_seconds
        self._model_cycle = cycle(self.models)

    def summarize_observation(self, evidence: dict[str, object]) -> SlmResponse:
        model = next(self._model_cycle)
        prompt = _observation_prompt(evidence)
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a concise SRE telemetry summarizer. Return only valid JSON with keys "
                        "summary, recommendation, confidence_reason, and risk."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 700,
        }
        started_at = time.perf_counter()
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except URLError as exc:
            raise RuntimeError(f"SLM request failed for {model}: {exc}") from exc
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        return SlmResponse(model=model, content=content, elapsed_seconds=time.perf_counter() - started_at)

    def extract_log_entry(self, evidence: dict[str, object]) -> SlmResponse:
        model = next(self._model_cycle)
        prompt = _log_entry_prompt(evidence)
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract compact SRE metadata from one telemetry log entry. Return only valid JSON with keys "
                        "summary, entities, signal_type, recommendation, confidence_reason, and risk."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 450,
        }
        started_at = time.perf_counter()
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except URLError as exc:
            raise RuntimeError(f"SLM request failed for {model}: {exc}") from exc
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        return SlmResponse(model=model, content=content, elapsed_seconds=time.perf_counter() - started_at)


def parse_slm_json(content: str) -> dict[str, object]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def _observation_prompt(evidence: dict[str, object]) -> str:
    return (
        "Summarize this telemetry evidence for an engineer. Be specific and avoid inventing facts. "
        "Use the model score, active alerts, top metric drivers, and mount/device context. "
        "Return compact JSON only. Keep each value under 45 words. "
        "Schema: {\"summary\":\"...\",\"recommendation\":\"...\",\"confidence_reason\":\"...\",\"risk\":\"low|medium|high\"}.\n\n"
        f"{json.dumps(evidence, indent=2, default=str)}"
    )


def _log_entry_prompt(evidence: dict[str, object]) -> str:
    return (
        "Extract metadata for this single telemetry log entry. Be specific, do not invent missing facts, "
        "and keep each textual field under 35 words. "
        "Schema: {\"summary\":\"...\",\"entities\":{\"instance\":\"...\",\"alertname\":\"...\",\"model\":\"...\","
        "\"metric\":\"...\",\"mountpoint\":\"...\",\"device\":\"...\"},\"signal_type\":\"alert|anomaly_score|slm_observation|other\","
        "\"recommendation\":\"...\",\"confidence_reason\":\"...\",\"risk\":\"low|medium|high\"}.\n\n"
        f"{json.dumps(evidence, indent=2, default=str)}"
    )

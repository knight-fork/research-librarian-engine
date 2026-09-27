"""Optional LLM providers behind one call: complete_json(system, user, schema) -> dict | None.

Providers: gemini (REST, model 'latest-flash-lite' resolves to the newest stable gemini-*-flash-lite)
and anthropic (official SDK). The LLM never supplies bibliographic metadata; callers validate its output."""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Dict, Optional, Tuple

from src.config import CACHE_DIR, Config
from src.http import HttpClient, HttpError

log = logging.getLogger("zotero_tool.llm")

GEMINI_API = "https://generativelanguage.googleapis.com/v1beta"
_STABLE_FLASH_LITE = re.compile(r"^models/gemini-(\d+(?:\.\d+)*)-flash-lite$")


class LLM:
    """Base: counts calls / tokens and enforces llm.max_calls_per_run."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.max_calls = int(cfg.get("llm.max_calls_per_run", 40))
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model = ""
        self._lock = threading.Lock()

    def _budget_ok(self) -> bool:
        with self._lock:
            if self.calls >= self.max_calls:
                return False
            self.calls += 1
            return True

    def complete_json(self, system: str, user: str, schema: Dict[str, Any], max_output_tokens: int = 4096) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def usage_note(self) -> str:
        return f"LLM {self.model}: {self.calls} call(s), {self.input_tokens} input / {self.output_tokens} output tokens"


class GeminiLLM(LLM):
    def __init__(self, cfg: Config, api_key: str, http: Optional[HttpClient] = None):
        super().__init__(cfg)
        self.api_key = api_key
        self.http = http or HttpClient(CACHE_DIR, timeout=120)
        self.model = self._resolve_model(str(cfg.get("llm.model", "latest-flash-lite")))

    def _headers(self) -> Dict[str, str]:
        return {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

    def _resolve_model(self, wanted: str) -> str:
        if wanted not in ("latest-flash-lite", "auto"):
            return wanted.replace("models/", "")
        try:
            data = self.http.get(f"{GEMINI_API}/models", params={"pageSize": 1000}, headers=self._headers(),
                                 use_cache=True, cache_ttl=24 * 3600)
        except HttpError as exc:
            log.warning("could not list Gemini models (%s); using gemini-flash-lite-latest", exc)
            return "gemini-flash-lite-latest"
        best: Tuple[Tuple[int, ...], str] = ((), "")
        for m in data.get("models") or []:
            mm = _STABLE_FLASH_LITE.match(m.get("name", ""))
            if mm and "generateContent" in (m.get("supportedGenerationMethods") or []):
                version = tuple(int(x) for x in mm.group(1).split("."))
                if version > best[0]:
                    best = (version, m["name"].replace("models/", ""))
        return best[1] or "gemini-flash-lite-latest"

    def complete_json(self, system, user, schema, max_output_tokens=4096):
        if not self._budget_ok():
            log.info("LLM call budget (%d) exhausted", self.max_calls)
            return None
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": schema,
                                 "temperature": 0, "maxOutputTokens": max_output_tokens},
        }
        try:
            data = self.http.request("POST", f"{GEMINI_API}/models/{self.model}:generateContent", json_body=body,
                                     headers=self._headers(), max_retries=3)
        except HttpError as exc:
            log.warning("Gemini call failed: %s", exc)
            return None
        usage = data.get("usageMetadata") or {}
        with self._lock:
            self.input_tokens += int(usage.get("promptTokenCount") or 0)
            self.output_tokens += int(usage.get("candidatesTokenCount") or 0)
        cands = data.get("candidates") or []
        if not cands or cands[0].get("finishReason") not in (None, "STOP", "MAX_TOKENS"):
            log.warning("Gemini returned no usable candidate (%s)", cands[0].get("finishReason") if cands else "empty")
            return None
        text = "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts") or [])
        try:
            return json.loads(text)
        except ValueError:
            log.warning("Gemini output was not valid JSON (finish reason %s)", cands[0].get("finishReason"))
            return None


class AnthropicLLM(LLM):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        import anthropic  # optional dependency
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = str(cfg.get("llm.anthropic_model", "claude-opus-5"))

    def complete_json(self, system, user, schema, max_output_tokens=4096):
        if not self._budget_ok():
            return None
        a = self._anthropic
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=max(max_output_tokens, 1024), system=system,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": user}],
            )
        except a.RateLimitError as exc:
            log.warning("LLM rate limited: %s", exc)
            return None
        except a.APIStatusError as exc:
            log.warning("LLM API error %s: %s", exc.status_code, exc.message)
            return None
        except a.APIConnectionError as exc:
            log.warning("LLM connection error: %s", exc)
            return None
        with self._lock:
            self.input_tokens += getattr(resp.usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(resp.usage, "output_tokens", 0) or 0
        if resp.stop_reason == "refusal":
            return None
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except ValueError:
            return None


_INSTANCES: Dict[int, Optional[LLM]] = {}


def get_llm(cfg: Config, task: str) -> Optional[LLM]:
    """The configured provider if llm.enabled and llm.tasks.<task> are on and credentials exist, else None."""
    if not cfg.get("llm.enabled", False) or not cfg.get(f"llm.tasks.{task}", False):
        return None
    key = id(cfg)
    if key not in _INSTANCES:
        provider = str(cfg.get("llm.provider", "gemini")).lower()
        inst: Optional[LLM] = None
        try:
            if provider == "gemini":
                if cfg.secrets.gemini_api_key:
                    inst = GeminiLLM(cfg, cfg.secrets.gemini_api_key)
                else:
                    log.warning("llm.provider is gemini but GEMINI_API_KEY is not set")
            elif provider == "anthropic":
                inst = AnthropicLLM(cfg)
            else:
                log.warning("unknown llm.provider %r", provider)
        except ImportError as exc:
            log.warning("LLM provider %s unavailable: %s", provider, exc)
        _INSTANCES[key] = inst
    return _INSTANCES[key]

"""LLM Director: turns a scene description into a ScenePlan JSON.

Providers are called over plain HTTPS (stdlib only), so the project has no
SDK lock-in: Anthropic Claude, Google Gemini or OpenAI, chosen by env vars.
Every response is parsed and validated against the pydantic schema; on
failure the validation errors are sent back for a bounded repair attempt.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from pydantic import ValidationError

from .library import MotionLibrary
from .schema import JSON_SCHEMA_HINT, ScenePlan

DEFAULT_MODELS = {
    "anthropic": os.getenv("SCENEFORGE_ANTHROPIC_MODEL", "claude-sonnet-5-5"),
    "gemini": os.getenv("SCENEFORGE_GEMINI_MODEL", "gemini-2.5-flash"),
    "openai": os.getenv("SCENEFORGE_OPENAI_MODEL", "gpt-4o-mini"),
}
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}

SYSTEM_PROMPT = """You are the Director Agent of SceneForge, an AI previsualization tool for a motion-capture studio.
Turn the user's scene description into a precise, shootable scene plan. Output ONLY one JSON object.

## Motion library (the ONLY clips that exist; use these ids in "clip")
{catalog}

## Rules
- Characters: ids "A", "B", "C"... (max 4). Give each a short name from the text (or a role like "Guard").
  If the user writes in Hindi, Telugu or another language, understand it and write the plan in English.
- Beats: one beat = one clip played by one character. Times in seconds from scene start.
  * Sequential actions of the same character must not overlap.
  * Reactions (dodge, fall, flinch) start 0.4-0.8 s after the attack they react to, so they overlap it.
  * "while", "as", "meanwhile" mean simultaneous beats.
  * Durations: locomotion 2-5 s (it loops), combat ~2.5 s, falls ~3 s, gestures 1.5-3.5 s.
  * Use "face" (a character id) when a character acts toward someone. Use "move_to" with
    {{"target": id, "distance": 1.0}} when a character walks or runs up to someone.
  * Pick the closest clip. If nothing fits, choose the nearest believable substitute and say so in "action".
  * speed: 0.8 for slow/careful, 1.2 for quick/sudden, else 1.0.
- Shots: cover the whole scene with contiguous shots (each 1.5-6 s). Follow the user's camera directions exactly
  (shot type, movement, lens). If they give none, design coverage like a real director: an establishing wide,
  then two-shots/mediums for interaction, close-ups on emotional or decisive moments, low angles for power,
  tracking for chases. Respect the 180-degree rule (keep the camera on one side of the action line).
- Keep total scene length under 40 s.

## Output JSON schema
{schema}
"""


@dataclass
class LLMResult:
    plan: ScenePlan | None
    provider: str
    model: str
    attempts: int = 0
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: list[str] = field(default_factory=list)
    raw: str = ""


def available_provider(preferred: str = "auto") -> str:
    preferred = (preferred or "auto").lower()
    if preferred == "offline":
        return "offline"
    if preferred in KEY_ENV:
        return preferred if os.getenv(KEY_ENV[preferred]) else "offline"
    for p in ("anthropic", "gemini", "openai"):
        if os.getenv(KEY_ENV[p]):
            return p
    return "offline"


def _post(url: str, body: dict, headers: dict, timeout: float = 60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300].decode(errors='ignore')}") from e


def _call(provider: str, model: str, system: str, messages: list[dict]) -> tuple[str, int, int]:
    """Returns (text, input_tokens, output_tokens)."""
    key = os.environ[KEY_ENV[provider]]
    if provider == "anthropic":
        d = _post("https://api.anthropic.com/v1/messages",
                  {"model": model, "max_tokens": 4096, "temperature": 0.4, "system": system, "messages": messages},
                  {"x-api-key": key, "anthropic-version": "2023-06-01"})
        text = "".join(b.get("text", "") for b in d.get("content", []))
        u = d.get("usage", {})
        return text, u.get("input_tokens", 0), u.get("output_tokens", 0)
    if provider == "gemini":
        contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                    for m in messages]
        d = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                  {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
                   "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json"}},
                  {"x-goog-api-key": key})
        text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
        u = d.get("usageMetadata", {})
        return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)
    if provider == "openai":
        d = _post("https://api.openai.com/v1/chat/completions",
                  {"model": model, "temperature": 0.4, "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": system}, *messages]},
                  {"authorization": f"Bearer {key}"})
        u = d.get("usage", {})
        return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    raise ValueError(provider)


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a model response (tolerates code fences / prose)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("no JSON object found in model output")


def plan_with_llm(prompt: str, lib: MotionLibrary, provider: str, model: str | None = None,
                  max_attempts: int = 3, caller=_call) -> LLMResult:
    model = model or DEFAULT_MODELS[provider]
    system = SYSTEM_PROMPT.format(catalog=lib.catalog_for_prompt(), schema=JSON_SCHEMA_HINT)
    messages = [{"role": "user", "content": f"Scene description:\n{prompt}"}]
    res = LLMResult(plan=None, provider=provider, model=model)
    t0 = time.perf_counter()
    for attempt in range(1, max_attempts + 1):
        res.attempts = attempt
        try:
            text, tin, tout = caller(provider, model, system, messages)
        except Exception as e:  # network/auth errors are not repairable by re-prompting
            res.errors.append(f"call failed: {e}")
            break
        res.input_tokens += tin
        res.output_tokens += tout
        res.raw = text
        try:
            data = extract_json(text)
            res.plan = ScenePlan.model_validate(data)
            break
        except (ValueError, ValidationError) as e:
            msg = str(e)[:1200]
            res.errors.append(f"attempt {attempt}: {msg.splitlines()[0]}")
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "Your JSON failed validation:\n" + msg +
                          "\nReturn the corrected, complete JSON object only."}]
    res.latency_ms = int((time.perf_counter() - t0) * 1000)
    return res

"""Thin OpenRouter helper: key from .env, concurrent chat calls, retries,
and a JSON file cache so re-runs never pay twice.

The key is read from the environment (``.env`` via python-dotenv) and is
never printed or logged.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import threading
import time

from dotenv import load_dotenv

_LOCK = threading.Lock()


def get_client():
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (expected in .env)")
    from openai import OpenAI
    # a hung request must not block a thread pool for the SDK default of 600 s
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=90.0, max_retries=1)


def _cache_key(model, messages, kwargs):
    h = hashlib.sha256(json.dumps([model, messages, kwargs], sort_keys=True).encode()).hexdigest()
    return h


def chat(client, model, messages, max_tokens=512, temperature=0.0, retries=8, cache=None, cache_path=None,
         extra_body=None, **kwargs):
    """One chat completion; returns the text. Cached by (model, messages, kwargs)."""
    ck = _cache_key(model, messages, dict(max_tokens=max_tokens, temperature=temperature, extra_body=extra_body, **kwargs))
    if cache is not None and ck in cache:
        return cache[ck]
    err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
                extra_body=extra_body or {}, **kwargs)
            text = resp.choices[0].message.content or ""
            if not text.strip():
                raise RuntimeError(f"empty completion from {model}")
            if cache is not None:
                with _LOCK:
                    cache[ck] = text
                    if cache_path:
                        _flush(cache, cache_path)
            return text
        except Exception as e:  # noqa: BLE001
            err = e
            # 402 = OpenRouter's in-flight budget is exhausted: wait for the in-flight requests to settle
            time.sleep(125 if "402" in str(e) or "in_flight" in str(e) else min(2 ** attempt, 30))
    raise RuntimeError(f"OpenRouter call failed after {retries} attempts: {err}")


def chat_many(client, model, message_lists, max_workers=16, **kw):
    """Concurrent ``chat`` over a list of message lists; preserves order."""
    out = [None] * len(message_lists)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(chat, client, model, m, **kw): i for i, m in enumerate(message_lists)}
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result()
    return out


def load_cache(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _flush(cache, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, path)

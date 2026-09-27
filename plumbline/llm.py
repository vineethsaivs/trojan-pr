"""Every LLM call goes through Vultr Serverless Inference (OpenAI-compatible). Each call returns a
record for the receipt: role, model, tokens, latency, sha256 of the raw response."""
import hashlib, logging, os, time
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

for _n in ("httpx", "openai"):
    logging.getLogger(_n).setLevel(logging.WARNING)

BASE = "https://api.vultrinference.com/v1"
_client = None


def client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=BASE, api_key=os.environ["VULTR_INFERENCE_API_KEY"], max_retries=0)
    return _client


def chat(role, model, messages, tools=None, tool_choice=None, max_tokens=4000, temperature=0, timeout=60,
         deadline=None):
    """-> (message, record). Retries 429, 5xx, timeouts and connection errors with backoff
    (rate limits are unpublished, VULTR.md), never past `deadline` (epoch seconds)."""
    kw = {"model": model, "messages": messages, "max_tokens": max_tokens, "timeout": timeout}
    if temperature is not None:                     # None = provider default (bake-off protocol)
        kw["temperature"] = temperature
    if tools:
        kw["tools"] = tools
    if tool_choice:
        kw["tool_choice"] = tool_choice
    last = None
    for attempt in range(4):
        t0 = time.time()
        if deadline:
            kw["timeout"] = min(timeout, deadline - t0)
            if kw["timeout"] < 3:
                raise last or TimeoutError(f"{role}: deadline reached")
        try:
            r = client().chat.completions.create(**kw)
            break
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            last = e
        except APIStatusError as e:
            if e.status_code < 500:
                raise
            last = e
        time.sleep(2 ** attempt)
    else:
        raise last
    u = r.usage
    rec = {"role": role, "model": model, "tokens_in": u.prompt_tokens if u else None,
           "tokens_out": u.completion_tokens if u else None, "latency_ms": int((time.time() - t0) * 1000),
           "response_sha256": hashlib.sha256(r.model_dump_json().encode()).hexdigest()}
    return r.choices[0].message, rec

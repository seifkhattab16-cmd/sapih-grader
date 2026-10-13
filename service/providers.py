"""AI providers, same order and defaults as the Supabase `ai` Edge Function. No Gemini (project rule).
Each provider is used only if its key is set. A provider that is rate-limited, down or rejects the
request is skipped and the next one is tried."""
from __future__ import annotations
from dataclasses import dataclass, field
import httpx
from . import config as C


class ProviderError(Exception):
    pass


@dataclass
class Route:
    name: str
    kind: str                      # "openai" | "anthropic"
    url: str
    key: str
    model: str
    max_images: int
    max_out: int
    headers: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    out_param: str = "max_tokens"


def build_routes(need_vision: bool, want_out: int = 3000) -> list[Route]:
    by: dict[str, list[Route]] = {"groq": [], "openrouter": [], "anthropic": []}
    degraded: list[Route] = []
    gk = C.env("GROQ_API_KEY")
    if gk:
        vision = C.env("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
        texts = C.lst(C.env("GROQ_TEXT_MODELS", "openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-20b"))

        def mk(model: str, imgs: int) -> Route:
            return Route(f"groq:{model}", "openai", "https://api.groq.com/openai/v1/chat/completions", gk, model,
                         imgs, min(want_out, 1500), out_param="max_completion_tokens",
                         extra={"reasoning_effort": "low"} if "gpt-oss" in model else {})
        if need_vision:
            by["groq"].append(mk(vision, 1))
            degraded += [mk(m, 0) for m in texts if m != vision]
        else:
            by["groq"] += [mk(m, 0) for m in texts]
    ok = C.env("OPENROUTER_API_KEY")
    if ok:
        for m in C.lst(C.env("OPENROUTER_MODELS", "inclusionai/ling-3.0-flash-sante:free,apodex/apodex-1.1-mini:free")):
            by["openrouter"].append(Route(f"openrouter:{m}", "openai", "https://openrouter.ai/api/v1/chat/completions",
                                          ok, m, 4, min(want_out, 3000), headers={"X-Title": "Saiph grader"}))
    ak = C.env("ANTHROPIC_API_KEY")
    if ak:
        by["anthropic"].append(Route("anthropic", "anthropic", "https://api.anthropic.com/v1/messages", ak,
                                     C.env("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"), 4, want_out))
    routes: list[Route] = []
    for p in C.route_order():
        routes += by.get(p, [])
    return routes + degraded


def _call(route: Route, system: str, text: str, images: list[str], client: httpx.Client) -> str:
    imgs = images[: route.max_images]
    if route.kind == "anthropic":
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}} for b in imgs]
        content.append({"type": "text", "text": text})
        r = client.post(route.url, headers={"x-api-key": route.key, "anthropic-version": "2023-06-01"},
                        json={"model": route.model, "max_tokens": route.max_out, "system": system,
                              "messages": [{"role": "user", "content": content}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []))
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in imgs]
    content.append({"type": "text", "text": text})
    body = {"model": route.model, route.out_param: route.max_out, **route.extra,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": content if imgs else text}]}
    r = client.post(route.url, headers={"Authorization": f"Bearer {route.key}", **route.headers}, json=body)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def complete(system: str, text: str, images: list[str] | None = None, want_out: int = 3000,
             routes: list[Route] | None = None, client: httpx.Client | None = None) -> tuple[str, str]:
    """Returns (reply text, provider name). Raises ProviderError when every route failed."""
    images = images or []
    routes = routes if routes is not None else build_routes(bool(images), want_out)
    if not routes:
        raise ProviderError("no AI provider key is set")
    own = client is None
    client = client or httpx.Client(timeout=120)
    errors = []
    try:
        for r in routes:
            try:
                out = _call(r, system, text, images, client)
                if out.strip():
                    return out, r.name
                errors.append(f"{r.name}: empty reply")
            except Exception as e:  # rate limit, outage, bad model name: try the next route
                errors.append(f"{r.name}: {type(e).__name__} {str(e)[:120]}")
        raise ProviderError("all providers failed: " + "; ".join(errors))
    finally:
        if own:
            client.close()

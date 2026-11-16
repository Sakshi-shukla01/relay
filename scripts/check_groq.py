"""Quick Groq connectivity check: is the key accepted, which models exist,
and what does one tiny call return? Prints no secrets.

  python scripts/check_groq.py
"""
import asyncio
import os
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from relay.config import DRAFT_MODEL, GROQ_BASE, ROUTER_MODEL  # noqa: E402


async def main():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("GROQ_API_KEY is not set in this terminal window.")
        return
    print(f"key: starts with {key[:4]!r}, length {len(key)}")
    h = {"Authorization": f"Bearer {key}"}
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{GROQ_BASE}/models", headers=h) as r:
            body = await r.text()
            print(f"\nGET /models -> HTTP {r.status}")
            if r.status != 200:
                print(body[:600])
                return
            import json
            ids = sorted(m["id"] for m in json.loads(body)["data"])
            print("models available:", ", ".join(ids))
            for m in (ROUTER_MODEL, DRAFT_MODEL):
                print(f"  {m}: {'available' if m in ids else 'NOT AVAILABLE'}")
        for m in (ROUTER_MODEL, DRAFT_MODEL):
            payload = {"model": m, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 5}
            async with s.post(f"{GROQ_BASE}/chat/completions", headers=h, json=payload) as r:
                print(f"\nPOST chat {m} -> HTTP {r.status}")
                print((await r.text())[:500])
                print("rate-limit headers:", {k: v for k, v in r.headers.items() if "ratelimit" in k.lower()})


if __name__ == "__main__":
    asyncio.run(main())

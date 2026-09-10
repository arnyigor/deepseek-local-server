"""Live probe: reasoning must work through the direct backend."""
import json
import sys
from pathlib import Path

import httpx

token = (Path.home() / "AppData/Local/deepseek-local-server/token").read_text().strip()
base = "http://127.0.0.1:9874"
headers = {"Authorization": f"Bearer {token}", "x-agent-session": "probe-reasoning"}

r = httpx.post(
    f"{base}/v1/chat/completions",
    headers=headers,
    json={"model": "deepseek-reasoner", "messages": [{"role": "user", "content": "Сколько будет 17*23? Ответь одним числом."}], "stream": False},
    timeout=120,
)
print("HTTP", r.status_code)
data = r.json()
if r.is_error:
    print(json.dumps(data, ensure_ascii=False))
    sys.exit(1)
msg = data["choices"][0]["message"]
backend = data.get("backend") or r.headers.get("x-deepseek-local-backend")
reasoning = msg.get("reasoning_content") or ""
content = msg.get("content") or ""
print("backend:", backend)
print("reasoning present:", bool(reasoning), "| len:", len(reasoning))
print("reasoning head:", reasoning[:150].replace("\n", " "))
print("content:", content[:150])
ok = bool(reasoning) and "391" in content
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

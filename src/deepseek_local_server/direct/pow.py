from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path

import httpx

from deepseek_local_server.errors import DirectProtocolError


class PowSolver:
    """Executes DeepSeek's official browser PoW WASM locally via wasmtime.

    The WASM URL is captured from the user's own logged-in DeepSeek browser session.
    Modules are cached on disk by URL hash and compiled once per process.
    """

    def __init__(self, cache_dir: Path, timeout_seconds: float = 15.0) -> None:
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds
        self._compiled: dict[str, object] = {}

    async def _load_bytes(self, url: str) -> bytes:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(url.encode("utf-8")).hexdigest() + ".wasm"
        path = self.cache_dir / name
        if path.exists():
            return path.read_bytes()
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.content
        path.write_bytes(data)
        return data

    async def solve(self, challenge: dict, wasm_url: str) -> int:
        try:
            import wasmtime
        except ImportError as exc:
            raise DirectProtocolError("wasmtime is required for the direct backend. Reinstall the project dependencies.") from exc

        module = self._compiled.get(wasm_url)
        if module is None:
            raw = await self._load_bytes(wasm_url)
            engine = wasmtime.Engine()
            module = (engine, wasmtime.Module(engine, raw))
            self._compiled[wasm_url] = module
        engine, compiled = module  # type: ignore[misc]
        store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        instance = linker.instantiate(store, compiled)
        exports = instance.exports(store)
        try:
            alloc = exports["__wbindgen_export_0"]
            stack = exports["__wbindgen_add_to_stack_pointer"]
            solve = exports["wasm_solve"]
            memory = exports["memory"]
        except KeyError as exc:
            raise DirectProtocolError(f"DeepSeek PoW WASM exports changed: missing {exc}") from exc

        prefix = f"{challenge['salt']}_{challenge['expire_at']}_".encode("utf-8")
        challenge_bytes = str(challenge["challenge"]).encode("utf-8")
        c_ptr = int(alloc(store, len(challenge_bytes), 1))
        p_ptr = int(alloc(store, len(prefix), 1))
        memory.write(store, challenge_bytes, c_ptr)
        memory.write(store, prefix, p_ptr)
        sp = int(stack(store, -16))
        solve(store, sp, c_ptr, len(challenge_bytes), p_ptr, len(prefix), float(challenge["difficulty"]))
        raw_result = bytes(memory.read(store, sp, sp + 16))
        stack(store, 16)
        code = struct.unpack_from("<i", raw_result, 0)[0]
        answer = struct.unpack_from("<d", raw_result, 8)[0]
        if code == 0 or not math.isfinite(answer) or answer <= 0:
            raise DirectProtocolError("DeepSeek PoW solver returned an invalid answer")
        return math.floor(answer)

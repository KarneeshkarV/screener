# Plan 006 — Reject cross-origin requests to the backtest-lab server

- **Status:** TODO
- **Written against commit:** `9547d4d` (re-verify excerpts if HEAD moved)
- **Category:** security (low severity — defense in depth)
- **Effort:** S · **Risk of fix:** low
- **Depends on:** nothing

## Why this matters

`backtest-lab` runs a plain `ThreadingHTTPServer` on `127.0.0.1:8766`. It validates payload *contents* (universe/strategy whitelists) but never checks **where the request came from**. Two consequences: any webpage open in the user's browser can fire `POST /api/run` at localhost (CSRF — browsers send it even cross-origin since the request is a "simple" POST), and a DNS-rebinding page can read responses too. Worst realistic impact is low (trigger backtests / burn CPU / read backtest results), but the fix is a few lines.

## Current state (verified excerpt)

`screener/backtester/lab.py:493-532` — `do_POST` parses and runs with no Origin/Host validation:

```python
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/run":
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(size)
            payload = json.loads(raw.decode() or "{}")
            ...
            data = compare_payload(...)
            self._send(HTTPStatus.OK, body, "application/json")
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode()
            self._send(HTTPStatus.BAD_REQUEST, body, "application/json")
```

Server startup at `lab.py:538-545` (`--host` default `127.0.0.1`, port 8766). `do_GET` at 483-491 serves the HTML UI and `/api/strategies`.

## The fix

Add a guard method to `LabHandler` and call it at the top of both `do_GET` and `do_POST`:

```python
    _ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

    def _same_origin(self) -> bool:
        """Reject DNS-rebinding (bad Host) and cross-site requests (bad Origin)."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if host not in self._ALLOWED_HOSTS:
            return False
        origin = self.headers.get("Origin")
        if origin:
            origin_host = urllib.parse.urlsplit(origin).hostname or ""
            if origin_host not in self._ALLOWED_HOSTS:
                return False
        return True
```

At the top of `do_GET` and `do_POST`:

```python
        if not self._same_origin():
            self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
            return
```

Details:

- `Host: 127.0.0.1:8766` → `rsplit(":", 1)[0]` gives `127.0.0.1`. IPv6 `[::1]:8766` → `[::1]`. Keep the bracket form in the allowlist as shown.
- The lab's own UI is served from the same host, so its fetches send `Origin: http://127.0.0.1:8766` (or no Origin for same-origin GET) — both pass.
- If the user passes a non-default `--host`, requests to that host name would be rejected by the static allowlist. Handle it: pass the configured host into the handler (e.g. set a class attribute `LabHandler.allowed_hosts = _ALLOWED_HOSTS | {host}` in `backtest_lab()` before constructing the server) so `--host 0.0.0.0` / LAN use keeps working. Look at how `backtest_lab` builds the server (`lab.py:541-545`) and keep the change minimal.
- `urllib.parse` may need importing — check the module's existing imports first.

## Tests

`tests/test_lab.py` exists — read it first to see whether it exercises the handler over HTTP or just `compare_payload`. Add handler-level tests by starting `ThreadingHTTPServer(("127.0.0.1", 0), LabHandler)` in a thread (port 0 = ephemeral; get it from `server.server_address`) and using `http.client` or `urllib.request`:

1. `GET /api/strategies` with default headers → 200.
2. `POST /api/run` with `Origin: http://evil.example` → 403, and the response body does not contain backtest data.
3. Request with `Host: evil.example` → 403.
4. `POST /api/run` with `Origin: http://127.0.0.1:<port>` and an intentionally invalid payload → 400 (proves the guard passes same-origin traffic through to existing logic; avoids running a real backtest in the test).

Shut the server down in a `finally:` (`server.shutdown(); server.server_close()`).

## Verification gates

```bash
uv run pytest tests/test_lab.py -q
uv run pytest -q
uv run ruff check $(git ls-files '*.py')
uv run ruff format --check $(git ls-files '*.py')
uv run mypy
```

Manual smoke (optional, requires a browser): `uv run screener backtest-lab`, open the printed URL, confirm the UI still loads and a run completes.

## Boundaries

- **In scope:** `screener/backtester/lab.py` (handler + the one line in `backtest_lab` wiring the allowed host), `tests/test_lab.py`.
- **Out of scope:** auth/tokens, HTTPS, changing default host/port, the HTML/JS UI, `compare_payload`, dashboard/tearsheet modules.

## Escape hatches

- If the lab UI's own fetches fail the guard in manual testing (unexpected Origin form), report the observed header rather than loosening the allowlist to a suffix match.

## Maintenance note

If a future change serves the lab over a LAN host or adds endpoints, every new handler method must call `_same_origin()` first — reviewers should check for it.

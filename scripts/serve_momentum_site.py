#!/usr/bin/env python
"""Serve the momentum study site over the local network.

    uv run python scripts/serve_momentum_site.py
    uv run python scripts/serve_momentum_site.py --port 8790

Binds every interface by default so the page is reachable from another machine
on the same Tailscale network, and prints the Tailscale URL when the CLI is
available. Pass ``--host 127.0.0.1`` to keep it on the loopback interface.

The server is read-only: it serves the generated ``site`` directory and nothing
above it.
"""

from __future__ import annotations

import argparse
import gzip
import io
import subprocess
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DEFAULT_SITE = Path("reports/momentum_study/site")
COMPRESSIBLE_SUFFIXES = (".json", ".html", ".css", ".js")
# Below roughly one packet's worth, compression costs more than it saves.
MIN_COMPRESS_BYTES = 1024


class _ReusableServer(ThreadingHTTPServer):
    # Without this a restart on the same port fails while the old socket is in
    # TIME_WAIT, which is the common case when iterating on the page.
    allow_reuse_address = True


class _GzipHandler(SimpleHTTPRequestHandler):
    """Static handler that gzips text payloads on the way out.

    The study index carries a row per run, so it grows with the sweep - past a
    couple of thousand runs it is several megabytes of JSON that every page load
    pulls down before anything renders. Over a tailnet that is the difference
    between an instant page and a visible wait, and JSON of this shape
    compresses roughly tenfold.
    """

    # Returns whatever the stdlib's own send_head returns - a readable body or
    # None - so the annotation stays as loose as the contract it implements.
    def send_head(self) -> Any:
        accepted = self.headers.get("Accept-Encoding", "")
        path = Path(self.translate_path(self.path))
        if (
            "gzip" not in accepted
            or path.suffix not in COMPRESSIBLE_SUFFIXES
            or not path.is_file()
            or path.stat().st_size < MIN_COMPRESS_BYTES
        ):
            return super().send_head()
        try:
            body = gzip.compress(path.read_bytes(), compresslevel=6)
        except OSError:
            return super().send_head()
        self.send_response(200)
        self.send_header("Content-type", self.guess_type(str(path)))
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Last-Modified", self.date_time_string(int(path.stat().st_mtime))
        )
        self.end_headers()
        # send_head's contract is to return a readable body for do_GET and let
        # do_HEAD discard it, so hand back the compressed bytes the same way.
        return io.BytesIO(body)


def tailscale_host() -> str | None:
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    address = result.stdout.strip().splitlines()
    return address[0].strip() if address else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args()

    site = args.site.resolve()
    if not (site / "index.html").exists():
        print(f"{site}/index.html is missing; run scripts/build_momentum_site.py first")
        return 1

    handler = partial(_GzipHandler, directory=str(site))
    with _ReusableServer((args.host, args.port), handler) as server:
        print(f"serving {site} on http://{args.host}:{args.port}")
        host = tailscale_host()
        if host:
            print(f"tailscale: http://{host}:{args.port}")
        print("Ctrl+C to stop")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

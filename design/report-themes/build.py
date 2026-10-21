"""Assemble each theme mockup from src/<theme>.html + the shared skeleton.

Inlines layout.css, sample-data.js and common.js so every output page is a
single self-contained file that opens from disk (charting libs load from CDN).
"""

from pathlib import Path

HERE = Path(__file__).parent
skeleton = (HERE / "_skeleton.html").read_text()
layout = (HERE / "layout.css").read_text()
data = (HERE / "sample-data.js").read_text()
common = (HERE / "common.js").read_text()

for src in sorted((HERE / "src").glob("*.html")):
    parts = dict(
        chunk.split("\n", 1) for chunk in src.read_text().split("@@ ")[1:]
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{parts["title"].strip()} - report theme</title>
{parts["head"].strip()}
  <style>
{parts["tokens"].strip()}
{layout}
{parts.get("css", "").strip()}
  </style>
</head>
<body>
{skeleton.replace("__LIB__", parts["lib"].strip())}
<script>{data}</script>
<script>{common}</script>
<script>
{parts["charts"].strip()}
</script>
</body>
</html>
"""
    (HERE / f"{src.stem}.html").write_text(page)
    print("built", src.stem)

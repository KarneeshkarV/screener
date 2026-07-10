"""Shared HTML document scaffolding for self-contained report pages.

Several commands render a static, self-contained HTML page (tear-sheets,
dashboards, the strategy lab, the screen report, optimization reports). They all
hand-rolled the same ``<!doctype html> … <head> … <body>`` skeleton around their
own CSS and body markup. :func:`html_page` owns that skeleton so each site only
supplies the parts that differ: its stylesheet, its body, and whether it needs
the responsive viewport meta tag or extra ``<head>`` markup (e.g. an inlined
Plotly ``<script>``).
"""

from __future__ import annotations

_VIEWPORT_META = (
    '\n  <meta name="viewport" content="width=device-width, initial-scale=1">'
)


def html_page(
    title: str,
    css: str,
    body: str,
    *,
    head_extra: str = "",
    viewport: bool = True,
) -> str:
    """Wrap page ``css`` and ``body`` in the shared HTML document shell.

    ``title`` is inserted verbatim into ``<title>`` (callers escape it if the
    value is untrusted). ``css`` is the stylesheet body only, without ``<style>``
    tags. ``body`` is the full inner HTML of ``<body>``. ``head_extra`` injects
    additional ``<head>`` markup after the title (e.g. an inlined script) and
    ``viewport`` toggles the responsive meta tag.
    """
    viewport_meta = _VIEWPORT_META if viewport else ""
    head_block = f"\n  {head_extra}" if head_extra else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">{viewport_meta}
  <title>{title}</title>{head_block}
  <style>
{css}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


__all__ = ["html_page"]

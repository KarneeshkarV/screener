"""Shared HTML document scaffolding for self-contained report pages.

Several commands render a static, self-contained HTML page (tear-sheets,
dashboards, the strategy lab, the screen report, optimization reports). They all
hand-rolled the same ``<!doctype html> … <head> … <body>`` skeleton around their
own CSS and body markup. :func:`html_page` owns that skeleton so each site only
supplies the parts that differ: its stylesheet, its body, and whether it needs
the responsive viewport meta tag or extra ``<head>`` markup.

Every page is themed: the shared Slate tokens and base styles from
:mod:`screener.report_theme` are emitted ahead of the page's own ``css``, and
``charts=True`` inlines the ECharts bundle plus the theme runtime that renders
every :func:`screener.report_theme.chart_html` container on the page.
"""

from __future__ import annotations

from screener.report_theme import THEME_CSS, chart_scripts

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
    charts: bool = False,
) -> str:
    """Wrap page ``css`` and ``body`` in the shared HTML document shell.

    ``title`` is inserted verbatim into ``<title>`` (callers escape it if the
    value is untrusted). ``css`` is the stylesheet body only, without ``<style>``
    tags. ``body`` is the full inner HTML of ``<body>``. ``head_extra`` injects
    additional ``<head>`` markup after the title (e.g. an inlined script) and
    ``viewport`` toggles the responsive meta tag. ``charts`` inlines the
    chart runtime at the end of ``<body>`` (after the chart containers).
    """
    viewport_meta = _VIEWPORT_META if viewport else ""
    head_block = f"\n  {head_extra}" if head_extra else ""
    scripts = f"\n{chart_scripts()}" if charts else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">{viewport_meta}
  <title>{title}</title>{head_block}
  <style>
{THEME_CSS}
{css}
  </style>
</head>
<body>
{body}{scripts}
</body>
</html>
"""


__all__ = ["html_page"]

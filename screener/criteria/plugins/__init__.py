"""Plugin modules for screening criteria.

Criteria are grouped by theme (``technical``, ``fundamental``); each is
registered by a top-level ``@criterion("name")`` decorator. New criteria are
added by decorating a function here and importing its module from
``screener.criteria._register_plugins``.
"""

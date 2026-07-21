"""Plugin directory for indicators.

Each ``*.py`` file registers an indicator via a top-level ``@indicator("name")``
decorator. New indicators are added by dropping a module here and importing it
from ``screener.indicators.registry._register_plugins``.
"""

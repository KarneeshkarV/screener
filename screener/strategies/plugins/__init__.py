"""Plugin directory for strategies.

Each ``*.py`` file registers a strategy via a top-level ``@strategy("name")``
decorator (callable strategies) or a ``register_expression_strategy(...)`` call
(expression strategies). New strategies are added by dropping a module here and
importing it from ``screener.strategies.spec.discover_plugins``.
"""

"""Plugin directory for strategies. Auto-discovered by ``screener.strategies``.

Drop a new ``*.py`` file in here with ``@strategy("name")`` for callable
strategies or a top-level ``register_expression_strategy(...)`` call for
expression strategies.
decorator — it will be registered automatically the next time the package
loads. No central edits required.
"""

# rules/__init__.py
"""
Project-local rules for HumAID experiments.

Keep all evolving rules here (e.g., RULES_1..RULES_10..)
"""
from .humaid_rules import (
    RULES_BASELINE,
    RULES_1,
    RULES_2,
    RULES_3,
    RULES_4,
    RULES_REGISTRY,
    get_rule,
)

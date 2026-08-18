"""Chemical reaction predictor core package."""

from .formula import parse_formula, molar_mass, FormulaError
from .catalog import find_by_name, get_by_formula, all_entries
from .resolver import resolve, ResolveError
from .templates import TEMPLATES, all_templates
from .engine import predict, compute_reaction, find_matches, EngineError, NoReactionError

__all__ = [
    'parse_formula', 'molar_mass', 'FormulaError',
    'find_by_name', 'get_by_formula', 'all_entries',
    'resolve', 'ResolveError',
    'TEMPLATES', 'all_templates',
    'predict', 'compute_reaction', 'find_matches', 'EngineError', 'NoReactionError',
]
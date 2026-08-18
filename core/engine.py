"""Reaction engine: match reactants, apply temperature gates, do stoichiometry."""

from .catalog import get_by_formula
from .formula import molar_mass, FormulaError
from .resolver import resolve, ResolveError
from .templates import TEMPLATES

class EngineError(ValueError):
    """Raised for invalid inputs or unresolvable reactants."""


class NoReactionError(EngineError):
    """Raised when no known reaction matches the given reactants + temperature."""


def _formula_mass(formula, fallback=None):
    try:
        return molar_mass(formula)
    except FormulaError:
        return fallback


def _formula_display(formula):
    """Pretty subscript rendering for HTML-safe display."""
    out = []
    i = 0
    while i < len(formula):
        ch = formula[i]
        if ch.isdigit():
            out.append('<sub>{}</sub>'.format(ch))
            i += 1
            continue
        if ch == '(':
            j = i
            while j < len(formula) and formula[j] != ')':
                j += 1
            if j < len(formula):
                j += 1
                out.append(formula[i:j])
                i = j
                continue
        out.append(ch)
        i += 1
    return ''.join(out)


def _balanced_equation(template, display=False):
    lhs = []
    for formula, coef in template['reactants']:
        lhs.append((str(coef) if coef > 1 else '') + (formula if not display else _formula_display(formula)))
    rhs = []
    for formula, coef in template['products']:
        rhs.append((str(coef) if coef > 1 else '') + (formula if not display else _formula_display(formula)))
    return ' + '.join(lhs) + ' -> ' + ' + '.join(rhs)


def find_matches(resolved, temperature):
    """Return templates matching the given resolved reactants, subject to temp gate.

    resolved: list of dicts with 'formula' key.
    temperature: float deg C (or None).
    """
    input_set = {r['formula'] for r in resolved}

    candidates = []
    for t in TEMPLATES:
        required = set(spec for spec, _ in t['reactants'])
        if not required.issubset(input_set):
            continue
        optional = set(t.get('optional', []))
        extra = input_set - required
        if not extra.issubset(optional):
            continue
        candidates.append(t)

    candidates.sort(key=lambda t: t.get('priority', 0), reverse=True)

    if temperature is None:
        return candidates

    feasible = []
    for t in candidates:
        cond = t.get('conditions', {})
        min_temp = cond.get('min_temp')
        max_temp = cond.get('max_temp')
        if min_temp is not None and temperature < min_temp:
            continue
        if max_temp is not None and temperature > max_temp:
            continue
        feasible.append(t)
    return feasible


def compute_reaction(resolved, temperature=None):
    """Run the full engine on resolved reactants. Returns a result dict."""
    if not resolved:
        raise EngineError('Provide at least one reactant.')

    matches = find_matches(resolved, temperature)

    if not matches:
        names = ', '.join(r['name'] for r in resolved)
        if temperature is not None and any(
                _below_min(resolved, t, temperature) for t in TEMPLATES if _template_matches(t, resolved)):
            raise NoReactionError(
                f"No known reaction for {names} at {temperature:g} C — conditions too cold for the matching reaction.")
        raise NoReactionError(f"No known reaction in the catalog for: {names}.")

    template = matches[0]

    # --- stoichiometry with limiting reagent ---
    moles = {}
    masses = {}
    for r in resolved:
        mm = r.get('molar_mass') or _formula_mass(r['formula'])
        if not mm:
            raise EngineError(f"Cannot determine molar mass of {r['formula']}")
        mass = r.get('mass_g')
        masses[r['formula']] = mass
        moles[r['formula']] = mass / mm if mass is not None else None

    reactant_specs = template['reactants']
    coefs = {f: c for f, c in reactant_specs}

    # limiting reagent: smallest moles/coef across all consumed reactants
    ratios = {}
    for f, c in coefs.items():
        m = moles.get(f)
        if m is None:
            continue
        ratios[f] = m / c

    limiting = min(ratios, key=ratios.get) if ratios else None
    limiting_ratio = ratios.get(limiting, 0.0)
    limiting_names = [f for f, r in ratios.items() if abs(r - limiting_ratio) < 1e-9]

    products = []
    for formula, coef in template['products']:
        mm = _formula_mass(formula)
        if mm is None:
            mm = 0.0
        amount_mol = limiting_ratio * coef
        products.append({
            'formula': formula,
            'name': _product_name(formula),
            'moles': round(amount_mol, 6),
            'grams': round(amount_mol * mm, 4),
        })

    leftover = []
    for formula, coef in coefs.items():
        if formula in limiting_names:
            continue
        used = limiting_ratio * coef
        remaining_mol = moles[formula] - used
        if remaining_mol > 1e-9:
            mm = _formula_mass(formula) or 0.0
            leftover.append({
                'formula': formula,
                'name': _product_name(formula),
                'moles': round(remaining_mol, 6),
                'grams': round(remaining_mol * mm, 4),
            })

    cond = template.get('conditions', {})
    return {
        'matched': True,
        'template_id': template['id'],
        'reaction_name': template['name'],
        'reaction_type': template['type'],
        'equation': _balanced_equation(template),
        'equation_display': _balanced_equation(template, display=True),
        'notes': template.get('notes', ''),
        'temperature_gate': {
            'min_temp': cond.get('min_temp'),
            'max_temp': cond.get('max_temp'),
        },
        'limiting_reagent': limiting,
        'limiting_reagent_name': _product_name(limiting) if limiting else None,
        'limiting_ratio': round(limiting_ratio, 6),
        'products': products,
        'leftover': leftover,
        'description': None,
    }


def _below_min(resolved, template, temperature):
    required = set(spec for spec, _ in template['reactants'])
    input_set = {r['formula'] for r in resolved}
    if required != input_set:
        return False
    cond = template.get('conditions', {})
    min_temp = cond.get('min_temp')
    return min_temp is not None and temperature < min_temp


def _template_matches(template, resolved):
    required = set(spec for spec, _ in template['reactants'])
    input_set = {r['formula'] for r in resolved}
    return required == input_set


def _product_name(formula):
    entry = get_by_formula(formula)
    if entry:
        return entry['name']
    # best-effort human name for formulas not in catalog
    from .catalog import _BY_FORMULA
    if formula in _BY_FORMULA:
        return _BY_FORMULA[formula]['name']
    return formula


def predict(reactants, temperature=None, allow_pubchem=True):
    """Top-level API. reactants: list of {'name': str, 'mass_g': float}.

    Returns a result dict (see compute_reaction) or raises EngineError/NoReactionError.
    """
    if not reactants:
        raise EngineError('Provide at least one reactant.')
    resolved = []
    for r in reactants:
        name = (r.get('name') or '').strip()
        if not name:
            raise EngineError('Every reactant needs a name.')
        try:
            entry = resolve(name, allow_pubchem=allow_pubchem)
        except ResolveError as e:
            raise EngineError(str(e))
        mass = r.get('mass_g')
        if mass is None or mass <= 0:
            raise EngineError(f"Provide a positive mass (g) for '{name}'.")
        mm = entry.get('molar_mass') or _formula_mass(entry['formula'])
        if not mm:
            mm = entry.get('molar_mass', None)
        resolved.append({
            'formula': entry['formula'],
            'name': entry['name'],
            'phase': entry.get('phase'),
            'category': entry.get('category'),
            'molar_mass': mm,
            'mass_g': float(mass),
            'source': entry.get('source', 'local'),
        })
    resolved = _aggregate(resolved)
    return compute_reaction(resolved, temperature=temperature)


def _aggregate(resolved):
    """Merge entries that share a formula, summing their masses."""
    by_formula = {}
    for r in resolved:
        if r['formula'] in by_formula:
            by_formula[r['formula']]['mass_g'] += r['mass_g']
        else:
            by_formula[r['formula']] = dict(r)
    return list(by_formula.values())
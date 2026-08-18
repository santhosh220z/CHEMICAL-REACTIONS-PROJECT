"""Reaction engine: match reactants, apply temperature windows, do stoichiometry.

Matching is lenient: a template fires when all of its required species are
present. Extra inputs (solvents, catalysts, additional chemicals) do not block
it — they are reported as "not involved" for that reaction. All in-range
matches are returned so the UI can show every reaction the inputs allow.
"""

from .catalog import get_by_formula
from .formula import molar_mass, FormulaError
from .resolver import resolve, ResolveError
from .templates import TEMPLATES

# Species that are almost always inert carriers/solvents and never "consumed"
# unless a template explicitly uses them as a reactant.
INERT_SPECIES = {'H2O'}


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
    """Return all templates whose required species are present in resolved.

    resolved: list of dicts with 'formula' key.
    temperature: float deg C (or None). When provided, only templates whose
    temperature window contains it are returned.
    """
    input_set = {r['formula'] for r in resolved}

    candidates = []
    for t in TEMPLATES:
        required = set(spec for spec, _ in t['reactants'])
        if not required.issubset(input_set):
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


def _temperature_status(template, temperature):
    """Return 'ok', 'cold', or 'hot' for a template at a given temperature."""
    cond = template.get('conditions', {})
    min_temp = cond.get('min_temp')
    max_temp = cond.get('max_temp')
    if min_temp is not None and temperature < min_temp:
        return 'cold'
    if max_temp is not None and temperature > max_temp:
        return 'hot'
    return 'ok'


def _input_amounts(resolved):
    """Map formula -> dict(moles, grams, name) for all resolved inputs."""
    amounts = {}
    for r in resolved:
        mm = r.get('molar_mass') or _formula_mass(r['formula'])
        if not mm:
            raise EngineError(f"Cannot determine molar mass of {r['formula']}")
        mass = r.get('mass_g') or 0.0
        amounts[r['formula']] = {
            'name': r['name'],
            'moles': mass / mm,
            'grams': mass,
        }
    return amounts


def _compute_single(template, inputs):
    """Compute stoichiometry for one template. Returns a reaction dict."""
    coefs = {f: c for f, c in template['reactants']}

    ratios = {}
    for f, c in coefs.items():
        m = inputs.get(f, {}).get('moles')
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
        remaining_mol = inputs[formula]['moles'] - used
        if remaining_mol > 1e-9:
            mm = _formula_mass(formula) or 0.0
            leftover.append({
                'formula': formula,
                'name': _product_name(formula),
                'moles': round(remaining_mol, 6),
                'grams': round(remaining_mol * mm, 4),
            })

    consumed = set(coefs)
    not_involved = []
    for formula, amount in inputs.items():
        if formula not in consumed and formula not in INERT_SPECIES:
            not_involved.append({
                'formula': formula,
                'name': amount['name'],
                'moles': round(amount['moles'], 6),
                'grams': round(amount['grams'], 4),
            })

    cond = template.get('conditions', {})
    return {
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
        'not_involved': not_involved,
    }


def _product_summary(reactions):
    """Aggregate products across reactions by formula."""
    by_formula = {}
    for reaction in reactions:
        for p in reaction['products']:
            entry = by_formula.setdefault(p['formula'], {
                'name': p['name'],
                'formula': p['formula'],
                'produced_by': [],
                'moles': 0.0,
                'grams': 0.0,
            })
            entry['produced_by'].append(reaction['reaction_name'])
            entry['moles'] += p['moles']
            entry['grams'] += p['grams']
    summary = []
    for entry in by_formula.values():
        entry['produced_by'] = sorted(set(entry['produced_by']))
        entry['moles'] = round(entry['moles'], 6)
        entry['grams'] = round(entry['grams'], 4)
        summary.append(entry)
    summary.sort(key=lambda e: e['grams'], reverse=True)
    return summary


def compute_reaction(resolved, temperature=None):
    """Run the full engine on resolved reactants. Returns a result dict."""
    if not resolved:
        raise EngineError('Provide at least one reactant.')

    inputs = _input_amounts(resolved)

    all_candidates = find_matches(resolved, None)
    in_range = find_matches(resolved, temperature) if temperature is not None else all_candidates

    if not in_range:
        names = ', '.join(r['name'] for r in resolved)
        if all_candidates:
            reasons = []
            for t in all_candidates[:3]:
                status = _temperature_status(t, temperature)
                cond = t.get('conditions', {})
                if status == 'cold':
                    reasons.append(f"'{t['name']}' needs at least {cond.get('min_temp')} C")
                elif status == 'hot':
                    reasons.append(f"'{t['name']}' is not stable above {cond.get('max_temp')} C")
                else:
                    reasons.append(f"'{t['name']}'")
            raise NoReactionError(
                f"No reaction occurs at {temperature:g} C with {names}. "
                f"Matching reaction(s) exist but are outside their temperature window: "
                + '; '.join(reasons) + '.')
        raise NoReactionError(f"No known reaction in the catalog for: {names}.")

    reactions = [_compute_single(t, inputs) for t in in_range]

    consumed_any = set()
    for t in in_range:
        consumed_any.update(spec for spec, _ in t['reactants'])

    unreacted = []
    for formula, amount in inputs.items():
        if formula not in consumed_any:
            unreacted.append({
                'formula': formula,
                'name': amount['name'],
                'moles': round(amount['moles'], 6),
                'grams': round(amount['grams'], 4),
            })

    return {
        'matched': True,
        'temperature': temperature,
        'reactions': reactions,
        'product_summary': _product_summary(reactions),
        'unreacted': unreacted,
        'description': None,
    }


def _product_name(formula):
    entry = get_by_formula(formula)
    if entry:
        return entry['name']
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
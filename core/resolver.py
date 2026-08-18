"""Resolve free-text chemical names to catalog entries, with a PubChem fallback."""

import re

from .catalog import find_by_name, get_by_formula

# Element name -> symbol, for formulas typed in words like "sodium chloride" handled by aliases,
# but also to support plain element names and simple lookup of unknowns.
ELEMENT_NAMES = {
    'hydrogen': 'H', 'helium': 'He', 'lithium': 'Li', 'beryllium': 'Be',
    'boron': 'B', 'carbon': 'C', 'nitrogen': 'N', 'oxygen': 'O',
    'fluorine': 'F', 'neon': 'Ne', 'sodium': 'Na', 'magnesium': 'Mg',
    'aluminum': 'Al', 'aluminium': 'Al', 'silicon': 'Si', 'phosphorus': 'P',
    'sulfur': 'S', 'sulphur': 'S', 'chlorine': 'Cl', 'argon': 'Ar',
    'potassium': 'K', 'calcium': 'Ca', 'scandium': 'Sc', 'titanium': 'Ti',
    'vanadium': 'V', 'chromium': 'Cr', 'manganese': 'Mn', 'iron': 'Fe',
    'cobalt': 'Co', 'nickel': 'Ni', 'copper': 'Cu', 'zinc': 'Zn',
    'gallium': 'Ga', 'germanium': 'Ge', 'arsenic': 'As', 'selenium': 'Se',
    'bromine': 'Br', 'krypton': 'Kr', 'rubidium': 'Rb', 'strontium': 'Sr',
    'yttrium': 'Y', 'zirconium': 'Zr', 'niobium': 'Nb', 'molybdenum': 'Mo',
    'technetium': 'Tc', 'ruthenium': 'Ru', 'rhodium': 'Rh', 'palladium': 'Pd',
    'silver': 'Ag', 'cadmium': 'Cd', 'indium': 'In', 'tin': 'Sn',
    'antimony': 'Sb', 'tellurium': 'Te', 'iodine': 'I', 'xenon': 'Xe',
    'cesium': 'Cs', 'caesium': 'Cs', 'barium': 'Ba', 'lanthanum': 'La',
    'gold': 'Au', 'mercury': 'Hg', 'thallium': 'Tl', 'lead': 'Pb',
    'bismuth': 'Bi', 'polonium': 'Po', 'radon': 'Rn', 'radium': 'Ra',
    'thorium': 'Th', 'uranium': 'U',
}


class ResolveError(ValueError):
    """Raised when a name cannot be resolved to a known chemical."""


def _normalize(name):
    return ' '.join(name.lower().split())


def _looks_like_formula(name):
    """Heuristic: is the input already a formula-like string (e.g. H2SO4, NaCl)?"""
    return bool(re.fullmatch(r'[A-Za-z][A-Za-z0-9]*(?:\([A-Za-z0-9]+\)\d*)*', name.strip()))


def resolve_local(name):
    """Try to resolve a name against the local catalog. Returns dict or None."""
    entry = find_by_name(name)
    if entry is not None:
        return dict(entry)
    return None


def resolve_pubchem(name):
    """Query PubChem REST for a compound's formula and molar mass. Returns dict or None."""
    import requests
    from .formula import molar_mass, FormulaError

    url = (
        'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/'
        f'{_quote(name)}/property/MolecularFormula,MolecularWeight/JSON'
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        props = data['PropertyTable']['Properties'][0]
        formula = props.get('MolecularFormula')
        if not formula:
            return None
        try:
            mm = molar_mass(formula)
        except FormulaError:
            mm = None
        return {
            'formula': formula,
            'name': props.get('IUPACName') or name.title(),
            'aliases': [name],
            'phase': 'unknown',
            'category': 'unknown',
            'molar_mass': mm,
            'source': 'pubchem',
        }
    except Exception:
        return None


def _quote(name):
    from urllib.parse import quote
    return quote(name.strip())


def resolve(name, allow_pubchem=True):
    """Resolve a chemical name to a catalog-style dict.

    Order: local catalog -> PubChem REST (optional) -> error.
    """
    entry = resolve_local(name)
    if entry is not None:
        entry['source'] = 'local'
        return entry
    if allow_pubchem:
        entry = resolve_pubchem(name)
        if entry is not None:
            return entry
    raise ResolveError(f"Could not identify the chemical '{name}'")

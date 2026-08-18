"""Periodic table data and molecular formula parsing utilities."""

ELEMENTS = {
    'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974,
    'S': 32.06, 'Cl': 35.45, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
    'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
    'Ga': 69.723, 'Ge': 72.630, 'As': 74.922, 'Se': 78.971, 'Br': 79.904,
    'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224,
    'Nb': 92.906, 'Mo': 95.95, 'Tc': 98, 'Ru': 101.07, 'Rh': 102.91,
    'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41, 'In': 114.82, 'Sn': 118.71,
    'Sb': 121.76, 'Te': 127.60, 'I': 126.90, 'Xe': 131.29, 'Cs': 132.91,
    'Ba': 137.33, 'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24,
    'Pm': 145, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93,
    'Dy': 162.50, 'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05,
    'Lu': 174.97, 'Hf': 178.49, 'Ta': 180.95, 'W': 183.84, 'Re': 186.21,
    'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59,
    'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Po': 209, 'At': 210,
    'Rn': 222, 'Fr': 223, 'Ra': 226, 'Ac': 227, 'Th': 232.04,
    'Pa': 231.04, 'U': 238.03, 'Np': 237, 'Pu': 244, 'Am': 243,
    'Cm': 247, 'Bk': 247, 'Cf': 251, 'Es': 252, 'Fm': 257,
    'Md': 258, 'No': 259, 'Lr': 262, 'Rf': 267, 'Db': 268,
    'Sg': 269, 'Bh': 270, 'Hs': 277, 'Mt': 278,
}

_ELEMENTS_TWO_LETTER = sorted((e for e in ELEMENTS if len(e) == 2), key=len, reverse=True)
_ELEMENTS_ONE_LETTER = sorted((e for e in ELEMENTS if len(e) == 1), key=len, reverse=True)


class FormulaError(ValueError):
    """Raised when a formula string cannot be parsed."""


def parse_formula(formula):
    """Parse a molecular formula string into a dict of element -> atom count.

    Handles parentheses and subscript numbers, e.g.:
      "H2SO4"     -> {'H': 2, 'S': 1, 'O': 4}
      "Ca(OH)2"   -> {'Ca': 1, 'O': 2, 'H': 2}
      "Fe2(SO4)3" -> {'Fe': 2, 'S': 3, 'O': 12}
      "H2O"       -> {'H': 2, 'O': 1}
    """
    if not formula:
        raise FormulaError('empty formula')
    counts, i = _parse_sequence(formula, 0)
    if i != len(formula):
        raise FormulaError(f"unbalanced parentheses in {formula}")
    return counts


def _parse_sequence(formula, i):
    """Parse elements/groups until ')' or end. Returns (element counts, next index)."""
    counts = {}
    n = len(formula)
    while i < n and formula[i] != ')':
        ch = formula[i]
        if ch == '(':
            group, i = _parse_sequence(formula, i + 1)
            if i >= n or formula[i] != ')':
                raise FormulaError(f"unbalanced parentheses in {formula}")
            i += 1  # consume ')'
            num = _read_number(formula, i)
            i += len(str(num)) if num > 1 else _digit_len(formula, i)
            for elem, count in group.items():
                counts[elem] = counts.get(elem, 0) + count * num
        elif ch.isdigit():
            raise FormulaError(f"unexpected number in {formula}")
        else:
            elem = _read_element(formula, i)
            counts[elem] = counts.get(elem, 0) + 1
            i += len(elem)
            num = _read_number(formula, i)
            if num > 1:
                counts[elem] += num - 1
                i += len(str(num))
    return counts, i


def _read_number(formula, i):
    start = i
    while i < len(formula) and formula[i].isdigit():
        i += 1
    if i == start:
        return 1
    return int(formula[start:i])


def _digit_len(formula, i):
    n = 0
    while i < len(formula) and formula[i].isdigit():
        i += 1
        n += 1
    return n


def _read_element(formula, i):
    for candidate in _ELEMENTS_TWO_LETTER:
        if formula[i:i + 2] == candidate:
            return candidate
    for candidate in _ELEMENTS_ONE_LETTER:
        if formula[i:i + 1] == candidate:
            return candidate
    raise FormulaError(f"unknown element near '{formula[i:i + 3]}' in {formula}")


def molar_mass(formula):
    """Compute the molar mass (g/mol) of a formula string."""
    counts = parse_formula(formula)
    total = 0.0
    for elem, count in counts.items():
        if elem not in ELEMENTS:
            raise FormulaError(f"no atomic mass for element {elem}")
        total += ELEMENTS[elem] * count
    return total


def formula_string(counts):
    """Render an element-count dict back into a formula string."""
    order = sorted(counts.items(), key=lambda kv: (-len(kv[0]), kv[0]))
    parts = []
    for elem, count in order:
        parts.append(elem)
        if count > 1:
            parts.append(str(count))
    return ''.join(parts)


def balance_element_counts(target_counts, scalar):
    """Scale an element-count dict by an integer and return normalized dict."""
    return {e: c * scalar for e, c in target_counts.items()}

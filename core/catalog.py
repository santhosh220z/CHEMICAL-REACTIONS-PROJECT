"""Chemical catalog: name -> formula, molar mass, and phase metadata."""

from .formula import molar_mass

# canonical key is the chemical formula (sorted formula string for easy matching)
# name aliases include common names, formulas as text, and case variants.
CATALOG = [
    # --- Acids ---
    {'formula': 'HCl', 'name': 'Hydrochloric Acid', 'aliases': ['hydrochloric acid', 'muratic acid', 'muriatic acid', 'hydrogen chloride'], 'phase': 'aq', 'category': 'acid'},
    {'formula': 'H2SO4', 'name': 'Sulfuric Acid', 'aliases': ['sulfuric acid', 'sulphuric acid', 'oil of vitriol'], 'phase': 'aq', 'category': 'acid'},
    {'formula': 'HNO3', 'name': 'Nitric Acid', 'aliases': ['nitric acid'], 'phase': 'aq', 'category': 'acid'},
    {'formula': 'H3PO4', 'name': 'Phosphoric Acid', 'aliases': ['phosphoric acid', 'phosphoric acid'], 'phase': 'aq', 'category': 'acid'},
    {'formula': 'CH3COOH', 'name': 'Acetic Acid', 'aliases': ['acetic acid', 'ethanoic acid', 'vinegar'], 'phase': 'aq', 'category': 'acid'},
    {'formula': 'H2CO3', 'name': 'Carbonic Acid', 'aliases': ['carbonic acid'], 'phase': 'aq', 'category': 'acid'},
    {'formula': 'H2S', 'name': 'Hydrogen Sulfide', 'aliases': ['hydrogen sulfide', 'hydrogen sulphide', 'hydrosulfuric acid'], 'phase': 'g', 'category': 'acid'},
    {'formula': 'HBr', 'name': 'Hydrobromic Acid', 'aliases': ['hydrobromic acid'], 'phase': 'aq', 'category': 'acid'},
    {'formula': 'HI', 'name': 'Hydroiodic Acid', 'aliases': ['hydroiodic acid', 'hydriodic acid'], 'phase': 'aq', 'category': 'acid'},

    # --- Bases / hydroxides ---
    {'formula': 'NaOH', 'name': 'Sodium Hydroxide', 'aliases': ['sodium hydroxide', 'caustic soda', 'lye'], 'phase': 'aq', 'category': 'base'},
    {'formula': 'KOH', 'name': 'Potassium Hydroxide', 'aliases': ['potassium hydroxide', 'caustic potash'], 'phase': 'aq', 'category': 'base'},
    {'formula': 'Ca(OH)2', 'name': 'Calcium Hydroxide', 'aliases': ['calcium hydroxide', 'slaked lime'], 'phase': 's', 'category': 'base'},
    {'formula': 'NH4OH', 'name': 'Ammonium Hydroxide', 'aliases': ['ammonium hydroxide', 'ammonia water'], 'phase': 'aq', 'category': 'base'},
    {'formula': 'Mg(OH)2', 'name': 'Magnesium Hydroxide', 'aliases': ['magnesium hydroxide', 'milk of magnesia'], 'phase': 's', 'category': 'base'},
    {'formula': 'Al(OH)3', 'name': 'Aluminum Hydroxide', 'aliases': ['aluminum hydroxide', 'aluminium hydroxide'], 'phase': 's', 'category': 'base'},
    {'formula': 'Ba(OH)2', 'name': 'Barium Hydroxide', 'aliases': ['barium hydroxide'], 'phase': 's', 'category': 'base'},

    # --- Oxides ---
    {'formula': 'H2O', 'name': 'Water', 'aliases': ['water', 'h2o', 'dihydrogen monoxide', 'dihydrogen oxide'], 'phase': 'l', 'category': 'solvent'},
    {'formula': 'CO2', 'name': 'Carbon Dioxide', 'aliases': ['carbon dioxide', 'co2'], 'phase': 'g', 'category': 'oxide'},
    {'formula': 'CO', 'name': 'Carbon Monoxide', 'aliases': ['carbon monoxide', 'co'], 'phase': 'g', 'category': 'oxide'},
    {'formula': 'SO2', 'name': 'Sulfur Dioxide', 'aliases': ['sulfur dioxide', 'sulphur dioxide', 'so2'], 'phase': 'g', 'category': 'oxide'},
    {'formula': 'SO3', 'name': 'Sulfur Trioxide', 'aliases': ['sulfur trioxide', 'sulphur trioxide', 'so3'], 'phase': 's', 'category': 'oxide'},
    {'formula': 'NO2', 'name': 'Nitrogen Dioxide', 'aliases': ['nitrogen dioxide', 'no2'], 'phase': 'g', 'category': 'oxide'},
    {'formula': 'N2O', 'name': 'Nitrous Oxide', 'aliases': ['nitrous oxide', 'nitrogen monoxide', 'laughing gas', 'n2o'], 'phase': 'g', 'category': 'oxide'},
    {'formula': 'NO', 'name': 'Nitric Oxide', 'aliases': ['nitric oxide', 'nitrogen(ii) oxide'], 'phase': 'g', 'category': 'oxide'},
    {'formula': 'Na2O', 'name': 'Sodium Oxide', 'aliases': ['sodium oxide'], 'phase': 's', 'category': 'oxide'},
    {'formula': 'CaO', 'name': 'Calcium Oxide', 'aliases': ['calcium oxide', 'quicklime', 'lime'], 'phase': 's', 'category': 'oxide'},
    {'formula': 'MgO', 'name': 'Magnesium Oxide', 'aliases': ['magnesium oxide', 'magnesia'], 'phase': 's', 'category': 'oxide'},
    {'formula': 'Fe2O3', 'name': 'Iron(III) Oxide', 'aliases': ['iron(iii) oxide', 'ferric oxide', 'iron oxide', 'rust'], 'phase': 's', 'category': 'oxide'},
    {'formula': 'FeO', 'name': 'Iron(II) Oxide', 'aliases': ['iron(ii) oxide', 'ferrous oxide'], 'phase': 's', 'category': 'oxide'},
    {'formula': 'CuO', 'name': 'Copper(II) Oxide', 'aliases': ['copper(ii) oxide', 'cupric oxide', 'copper oxide'], 'phase': 's', 'category': 'oxide'},
    {'formula': 'Cu2O', 'name': 'Copper(I) Oxide', 'aliases': ['copper(i) oxide', 'cuprous oxide'], 'phase': 's', 'category': 'oxide'},
    {'formula': 'ZnO', 'name': 'Zinc Oxide', 'aliases': ['zinc oxide'], 'phase': 's', 'category': 'oxide'},
    {'formula': 'Al2O3', 'name': 'Aluminum Oxide', 'aliases': ['aluminum oxide', 'aluminium oxide', 'alumina'], 'phase': 's', 'category': 'oxide'},
    {'formula': 'MnO2', 'name': 'Manganese Dioxide', 'aliases': ['manganese dioxide', 'manganese(iv) oxide'], 'phase': 's', 'category': 'oxide'},
    {'formula': 'PbO', 'name': 'Lead(II) Oxide', 'aliases': ['lead(ii) oxide', 'lead oxide'], 'phase': 's', 'category': 'oxide'},
    {'formula': 'SiO2', 'name': 'Silicon Dioxide', 'aliases': ['silicon dioxide', 'silica', 'sand'], 'phase': 's', 'category': 'oxide'},
    {'formula': 'H2O2', 'name': 'Hydrogen Peroxide', 'aliases': ['hydrogen peroxide', 'h2o2'], 'phase': 'aq', 'category': 'peroxide'},

    # --- Salts (chlorides, sulfates, nitrates, carbonates, etc.) ---
    {'formula': 'NaCl', 'name': 'Sodium Chloride', 'aliases': ['sodium chloride', 'table salt', 'salt', 'nacl'], 'phase': 's', 'category': 'salt'},
    {'formula': 'KCl', 'name': 'Potassium Chloride', 'aliases': ['potassium chloride'], 'phase': 's', 'category': 'salt'},
    {'formula': 'CaCl2', 'name': 'Calcium Chloride', 'aliases': ['calcium chloride'], 'phase': 's', 'category': 'salt'},
    {'formula': 'MgCl2', 'name': 'Magnesium Chloride', 'aliases': ['magnesium chloride'], 'phase': 's', 'category': 'salt'},
    {'formula': 'FeCl2', 'name': 'Iron(II) Chloride', 'aliases': ['iron(ii) chloride', 'ferrous chloride'], 'phase': 's', 'category': 'salt'},
    {'formula': 'FeCl3', 'name': 'Iron(III) Chloride', 'aliases': ['iron(iii) chloride', 'ferric chloride'], 'phase': 's', 'category': 'salt'},
    {'formula': 'CuCl2', 'name': 'Copper(II) Chloride', 'aliases': ['copper(ii) chloride', 'cupric chloride'], 'phase': 's', 'category': 'salt'},
    {'formula': 'ZnCl2', 'name': 'Zinc Chloride', 'aliases': ['zinc chloride'], 'phase': 's', 'category': 'salt'},
    {'formula': 'NH4Cl', 'name': 'Ammonium Chloride', 'aliases': ['ammonium chloride', 'sal ammoniac'], 'phase': 's', 'category': 'salt'},
    {'formula': 'KClO3', 'name': 'Potassium Chlorate', 'aliases': ['potassium chlorate', 'chlorate of potash'], 'phase': 's', 'category': 'salt'},
    {'formula': 'BaCl2', 'name': 'Barium Chloride', 'aliases': ['barium chloride'], 'phase': 's', 'category': 'salt'},
    {'formula': 'Na2SO4', 'name': 'Sodium Sulfate', 'aliases': ['sodium sulfate', 'sodium sulphate'], 'phase': 's', 'category': 'salt'},
    {'formula': 'MgSO4', 'name': 'Magnesium Sulfate', 'aliases': ['magnesium sulfate', 'magnesium sulphate', 'epsom salt'], 'phase': 's', 'category': 'salt'},
    {'formula': 'CaSO4', 'name': 'Calcium Sulfate', 'aliases': ['calcium sulfate', 'calcium sulphate', 'gypsum'], 'phase': 's', 'category': 'salt'},
    {'formula': 'CuSO4', 'name': 'Copper(II) Sulfate', 'aliases': ['copper(ii) sulfate', 'cupric sulfate', 'copper sulfate', 'blue vitriol'], 'phase': 's', 'category': 'salt'},
    {'formula': 'FeSO4', 'name': 'Iron(II) Sulfate', 'aliases': ['iron(ii) sulfate', 'ferrous sulfate', 'green vitriol'], 'phase': 's', 'category': 'salt'},
    {'formula': 'ZnSO4', 'name': 'Zinc Sulfate', 'aliases': ['zinc sulfate'], 'phase': 's', 'category': 'salt'},
    {'formula': 'BaSO4', 'name': 'Barium Sulfate', 'aliases': ['barium sulfate', 'barium sulphate'], 'phase': 's', 'category': 'salt'},
    {'formula': 'NaNO3', 'name': 'Sodium Nitrate', 'aliases': ['sodium nitrate', 'chile saltpeter'], 'phase': 's', 'category': 'salt'},
    {'formula': 'KNO3', 'name': 'Potassium Nitrate', 'aliases': ['potassium nitrate', 'saltpeter', 'saltpetre'], 'phase': 's', 'category': 'salt'},
    {'formula': 'AgNO3', 'name': 'Silver Nitrate', 'aliases': ['silver nitrate'], 'phase': 's', 'category': 'salt'},
    {'formula': 'NH4NO3', 'name': 'Ammonium Nitrate', 'aliases': ['ammonium nitrate'], 'phase': 's', 'category': 'salt'},
    {'formula': 'Ca(NO3)2', 'name': 'Calcium Nitrate', 'aliases': ['calcium nitrate'], 'phase': 's', 'category': 'salt'},
    {'formula': 'Na2CO3', 'name': 'Sodium Carbonate', 'aliases': ['sodium carbonate', 'soda ash', 'washing soda'], 'phase': 's', 'category': 'carbonate'},
    {'formula': 'K2CO3', 'name': 'Potassium Carbonate', 'aliases': ['potassium carbonate', 'potash'], 'phase': 's', 'category': 'carbonate'},
    {'formula': 'CaCO3', 'name': 'Calcium Carbonate', 'aliases': ['calcium carbonate', 'limestone', 'chalk', 'marble'], 'phase': 's', 'category': 'carbonate'},
    {'formula': 'MgCO3', 'name': 'Magnesium Carbonate', 'aliases': ['magnesium carbonate'], 'phase': 's', 'category': 'carbonate'},
    {'formula': 'NaHCO3', 'name': 'Sodium Bicarbonate', 'aliases': ['sodium bicarbonate', 'baking soda', 'sodium hydrogen carbonate'], 'phase': 's', 'category': 'carbonate'},
    {'formula': 'KHCO3', 'name': 'Potassium Bicarbonate', 'aliases': ['potassium bicarbonate', 'potassium hydrogen carbonate'], 'phase': 's', 'category': 'carbonate'},
    {'formula': 'NH4HCO3', 'name': 'Ammonium Bicarbonate', 'aliases': ['ammonium bicarbonate', 'ammonium hydrogen carbonate'], 'phase': 's', 'category': 'carbonate'},

    # --- Pure elements / gases ---
    {'formula': 'H2', 'name': 'Hydrogen Gas', 'aliases': ['hydrogen', 'hydrogen gas', 'h2', 'dihydrogen'], 'phase': 'g', 'category': 'element'},
    {'formula': 'O2', 'name': 'Oxygen Gas', 'aliases': ['oxygen', 'oxygen gas', 'o2', 'dioxygen'], 'phase': 'g', 'category': 'element'},
    {'formula': 'N2', 'name': 'Nitrogen Gas', 'aliases': ['nitrogen', 'nitrogen gas', 'n2', 'dinitrogen'], 'phase': 'g', 'category': 'element'},
    {'formula': 'Cl2', 'name': 'Chlorine Gas', 'aliases': ['chlorine', 'chlorine gas', 'cl2'], 'phase': 'g', 'category': 'element'},
    {'formula': 'F2', 'name': 'Fluorine Gas', 'aliases': ['fluorine', 'fluorine gas', 'f2'], 'phase': 'g', 'category': 'element'},
    {'formula': 'Br2', 'name': 'Bromine', 'aliases': ['bromine', 'bromine liquid', 'br2'], 'phase': 'l', 'category': 'element'},
    {'formula': 'I2', 'name': 'Iodine', 'aliases': ['iodine', 'iodine solid', 'i2'], 'phase': 's', 'category': 'element'},
    {'formula': 'S', 'name': 'Sulfur', 'aliases': ['sulfur', 'sulphur', 'sulfur powder'], 'phase': 's', 'category': 'element'},
    {'formula': 'P', 'name': 'Phosphorus', 'aliases': ['phosphorus', 'white phosphorus', 'red phosphorus'], 'phase': 's', 'category': 'element'},
    {'formula': 'C', 'name': 'Carbon', 'aliases': ['carbon', 'charcoal', 'graphite', 'diamond', 'soot'], 'phase': 's', 'category': 'element'},
    {'formula': 'Na', 'name': 'Sodium', 'aliases': ['sodium', 'sodium metal'], 'phase': 's', 'category': 'metal'},
    {'formula': 'K', 'name': 'Potassium', 'aliases': ['potassium', 'potassium metal'], 'phase': 's', 'category': 'metal'},
    {'formula': 'Li', 'name': 'Lithium', 'aliases': ['lithium', 'lithium metal'], 'phase': 's', 'category': 'metal'},
    {'formula': 'Ca', 'name': 'Calcium', 'aliases': ['calcium', 'calcium metal'], 'phase': 's', 'category': 'metal'},
    {'formula': 'Mg', 'name': 'Magnesium', 'aliases': ['magnesium', 'magnesium metal', 'magnesium ribbon'], 'phase': 's', 'category': 'metal'},
    {'formula': 'Al', 'name': 'Aluminum', 'aliases': ['aluminum', 'aluminium', 'aluminum metal'], 'phase': 's', 'category': 'metal'},
    {'formula': 'Zn', 'name': 'Zinc', 'aliases': ['zinc', 'zinc metal', 'zinc dust'], 'phase': 's', 'category': 'metal'},
    {'formula': 'Fe', 'name': 'Iron', 'aliases': ['iron', 'iron metal', 'steel', 'wrought iron'], 'phase': 's', 'category': 'metal'},
    {'formula': 'Cu', 'name': 'Copper', 'aliases': ['copper', 'copper metal'], 'phase': 's', 'category': 'metal'},
    {'formula': 'Ag', 'name': 'Silver', 'aliases': ['silver', 'silver metal'], 'phase': 's', 'category': 'metal'},
    {'formula': 'Au', 'name': 'Gold', 'aliases': ['gold', 'gold metal'], 'phase': 's', 'category': 'metal'},
    {'formula': 'Pt', 'name': 'Platinum', 'aliases': ['platinum', 'platinum metal'], 'phase': 's', 'category': 'metal'},
    {'formula': 'Ni', 'name': 'Nickel', 'aliases': ['nickel', 'nickel metal'], 'phase': 's', 'category': 'metal'},
    {'formula': 'Pb', 'name': 'Lead', 'aliases': ['lead', 'lead metal'], 'phase': 's', 'category': 'metal'},
    {'formula': 'Sn', 'name': 'Tin', 'aliases': ['tin', 'tin metal'], 'phase': 's', 'category': 'metal'},
    {'formula': 'Hg', 'name': 'Mercury', 'aliases': ['mercury', 'quicksilver', 'mercury metal'], 'phase': 'l', 'category': 'metal'},

    # --- Hydrocarbons / fuels for combustion ---
    {'formula': 'CH4', 'name': 'Methane', 'aliases': ['methane', 'natural gas', 'ch4'], 'phase': 'g', 'category': 'hydrocarbon'},
    {'formula': 'C2H6', 'name': 'Ethane', 'aliases': ['ethane', 'c2h6'], 'phase': 'g', 'category': 'hydrocarbon'},
    {'formula': 'C3H8', 'name': 'Propane', 'aliases': ['propane', 'c3h8'], 'phase': 'g', 'category': 'hydrocarbon'},
    {'formula': 'C4H10', 'name': 'Butane', 'aliases': ['butane', 'c4h10'], 'phase': 'g', 'category': 'hydrocarbon'},
    {'formula': 'C2H4', 'name': 'Ethylene', 'aliases': ['ethylene', 'ethene', 'c2h4'], 'phase': 'g', 'category': 'hydrocarbon'},
    {'formula': 'C2H5OH', 'name': 'Ethanol', 'aliases': ['ethanol', 'ethyl alcohol', 'alcohol', 'c2h5oh'], 'phase': 'l', 'category': 'hydrocarbon'},
    {'formula': 'C6H12O6', 'name': 'Glucose', 'aliases': ['glucose', 'dextrose', 'sugar'], 'phase': 's', 'category': 'organic'},

    # --- Ammonia ---
    {'formula': 'NH3', 'name': 'Ammonia', 'aliases': ['ammonia', 'ammonia gas', 'nh3'], 'phase': 'g', 'category': 'hydride'},
]


def _normalize_alias(alias):
    return ' '.join(alias.lower().split())


# Build lookup structures
_BY_FORMULA = {}
_BY_ALIAS = {}
for entry in CATALOG:
    _BY_FORMULA[entry['formula']] = entry
    for alias in entry['aliases']:
        _BY_ALIAS.setdefault(_normalize_alias(alias), entry)


def get_by_formula(formula):
    """Look up a catalog entry by its canonical formula string."""
    return _BY_FORMULA.get(formula)


def find_by_name(name):
    """Look up a catalog entry by an alias/name. Returns entry or None."""
    return _BY_ALIAS.get(_normalize_alias(name))


def all_entries():
    return list(CATALOG)


def formula_molar_mass(formula):
    from .formula import molar_mass
    return molar_mass(formula)

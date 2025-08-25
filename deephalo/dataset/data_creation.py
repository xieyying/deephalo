from molmass import Formula
import pandas as pd

from fake_isotope_generator import FakeIsotopeGenerator, mass_spectrum_calc


def create_data(formula, type='base', rate=None) -> pd.DataFrame:
    """
    Simulate mass spectrometry data based on the formula and its type.

    Args:
    formula: str, the chemical formula.
    type: str, the type of formula (e.g., 'base', 'Fe', 'B', 'Se', 'hydro').
    rate: float, the rate for hydrogenation (only used for 'hydro' type).

    Returns:
    pd.DataFrame: Simulated mass spectrometry data.
    """
    if not isinstance(formula, str):
        raise ValueError(formula, 'formula must be a string')
    
    # Convert the formula to a dictionary representation for classification
    formula_dict = Formula(formula).composition().dataframe().to_dict()['Count']
    
    # Determine if the formula is trainable and classify it
    trainable, group = formula_clf(formula_dict, type=type)

    if trainable == 'no':
        return pd.DataFrame()
    elif type in ['Fe', 'B', 'Se', 'hydro']:
        # Check if the formula contains only valid elements
        if not set(formula_dict.keys()).issubset(set(['C', 'H', 'O', 'N'])):
            return pd.DataFrame()
        
    # Simulate mass spectrometry data
    dict_isos = FakeIsotopeGenerator(formula, type, rate).generate(peak_n=10)

    if type == 'Se':
        group = 3
    elif type == 'B':
        group = 4
    elif type == 'Fe':
        group = 5
    
    dict_base = {'formula': formula, 'group': group}

    dict_isos_calc = mass_spectrum_calc(dict_isos, 1)

    # Merge dict_base, dict_isos, and dict_isos_calc
    dict_all = dict_base.copy()
    dict_all.update(dict_isos)
    dict_all.update(dict_isos_calc)
    df = pd.DataFrame([dict_all])

    return df


def formula_clf(formula_dict, type=None):
    """
    Returns a class based on the formula given.

    Args:
    formula_dict: dict, dictionary representation of the formula.
    type: str, the type of formula.

    Returns:
    trainable: str, whether the formula can be trained ('yes' or 'no').
    group: int, classification group of the formula.
    """
    # Determine if the formula is trainable
    if formula_dict.get('R') is not None:
        trainable = 'no'
    elif formula_dict.get('H') is None or formula_dict.get('C') is None:
        trainable = 'no'
    elif formula_dict.get('H') < 3 or formula_dict.get('C') < 1:
        trainable = 'no'
    elif 'S' in formula_dict.keys() and formula_dict.get('S') > 4:
        trainable = 'no'
    elif 'Se' in formula_dict.keys() and formula_dict.get('Se') > 1:
        trainable = 'no'
    else:
        trainable = 'yes'

    # Determine the classification group based on the formula
    if type == 'hydro':
        group = 7

    elif ('Br' in formula_dict.keys()) and ('Cl' in formula_dict.keys()):
        if 'B' in formula_dict.keys() or 'Se' in formula_dict.keys() or 'Fe' in formula_dict.keys():
            group = 10
        else:
            group = 0
    elif ('Br' in formula_dict.keys()) or ('Cl' in formula_dict.keys()):
        if 'B' in formula_dict.keys() or 'Se' in formula_dict.keys() or 'Fe' in formula_dict.keys():
            group = 10
        elif ('Br' in formula_dict.keys()) and formula_dict['Br'] > 1:
            group = 0
        elif ('Cl' in formula_dict.keys()) and formula_dict['Cl'] > 3:
            group = 0        
        elif ('Br' in formula_dict.keys()) and formula_dict['Br'] == 1:
            group = 1
        elif ('Cl' in formula_dict.keys()) and formula_dict['Cl'] == 3:
            group = 1
        elif ('Cl' in formula_dict.keys()) and formula_dict['Cl'] >= 1:
            group = 2
    elif 'Se' in formula_dict.keys():
        group = 3

    elif 'B' in formula_dict.keys():
        group = 4

    elif 'Fe' in formula_dict.keys():
        group = 5

    else:
        group = 6

    return trainable, group


if __name__ == '__main__':
    df = create_data('C30H15Br7O9S', 'base')
    print(df)
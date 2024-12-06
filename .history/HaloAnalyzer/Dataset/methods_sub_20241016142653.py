import numpy as np
from molmass import Formula,Spectrum

def get_iron_additive_isotopes(formula):
    """get the isotopic distribution of the formula with iron additive"""
    f=Formula(formula+"Fe")-Formula('H2')
    return f.spectrum()
def get_boron_additive_isotopes(formula):
    """get the isotopic distribution of the formula with boron additive"""
    f=Formula(formula+"B")-Formula('H3')
    return f.spectrum()
def get_selenium_additive_isotopes(formula):
    """get the isotopic distribution of the formula with selenium additive"""
    f=Formula(formula+"Se")
    return f.spectrum()



def mass_spectrum_calc(dict_features,charge) -> dict:
    """
    重新转换质谱数据的特征，使其以mz最小的峰为m0的质谱数据，同时计算相关特征

    Args:
    dict_features: dict, 质谱数据的特征
    charge: int, 电荷数

    Returns:
    dict, 质谱数据的相关特征
    """
    mz_2 = dict_features['mz_2']
    mz_1 = dict_features['mz_1']
    mz_0 = dict_features['mz_0']

    if mz_2 !=0:
        m2_m1 = (mz_2 - mz_1)*charge
        m2_m0 = (mz_2 - mz_0)*charge
    else:
        m2_m1 = 1.002
        m2_m0 = 2.002

    if mz_1 !=0:
        m1_m0 = (mz_1 - mz_0)*charge 
    else:    
        m1_m0 = 1.002

    dict_features['m2_m1'] = m2_m1
    dict_features['m2_m0'] = m2_m0
    dict_features['m1_m0'] = m1_m0


    #以字典的形式返回
    return dict_features

def get_hydroisomer_isotopes(formula, ratio, min_intensity=0.0001) -> Spectrum:
    """
    Calculate the overlapped spectrum of a chemical formula and its hydroisomer.
    
    Parameters:
    formula: str, the chemical formula.
    ratio: float, the weighting between the two spectra in the final overlapped spectrum.
    min_intensity: float, the minimum relative intensity for an isotope to be included in the final spectrum.

    Returns:
    Spectrum: the overlapped spectrum.
    """
    # Calculate the isotopic distribution of the original formula
    spectrum1 = Formula(formula).spectrum()
    # Calculate the isotopic distribution of the formula with two additional hydrogen atoms
    spectrum2 = Formula(formula + "H2").spectrum()

    min_fraction: float = 1e-16

    # Create a new dictionary to store the overlapped spectrum
    spectrum = {}
    for key1, items in sorted(spectrum1.items()):
        f = items.fraction
        m = items.mass
        k = items.massnumber
        if key1 in spectrum2:
            f2 = spectrum2[key1].fraction
            m2 = spectrum2[key1].mass
            # Calculate the new mass and relative intensity for the isotope
            m_new = (f * m + f2 * m2 * ratio) / (f + f2 * ratio)
            f_new = f + f2 * ratio
            spectrum[k] = [m_new, f_new, 1.0]
        else:
            # If the isotope is only present in spectrum 1, retain its original mass and relative intensity
            spectrum[k] = [m, f, 1.0]
    # Add isotopes that are exclusively present in spectrum2 to the overlapped spectrum
    for key2, items in sorted(spectrum2.items()):
        if key2 not in spectrum1:
            f = items.fraction
            m = items.mass
            k = items.massnumber
            spectrum[k] = [m, f, 1.0]

    # Filter out isotopes with low intensities and normalize the remaining isotopic intensities to 100
    if min_intensity is not None:
        norm = 100 / max(v[1] for v in spectrum.values())
        for massnumber, value in spectrum.copy().items():
            if value[1] * norm < min_intensity:
                del spectrum[massnumber]
            else:
                spectrum[massnumber][2] = value[1] * norm

    return Spectrum(spectrum)

def get_dehydroisomer_isotopes(formula, ratio, min_intensity=0.0001) -> Spectrum:
    """
    Calculate the overlapped spectrum of a chemical formula and its hydroisomer.
    
    Parameters:
    formula: str, the chemical formula.
    ratio: float, the weighting between the two spectra in the final overlapped spectrum.
    min_intensity: float, the minimum relative intensity for an isotope to be included in the final spectrum.

    Returns:
    Spectrum: the overlapped spectrum.
    """
    
    # Calculate the isotopic distribution of the formula with two additional hydrogen atoms
    spectrum1 = (Formula(formula) - Formula("H2")).spectrum()
    
    # Calculate the isotopic distribution of the original formula
    spectrum2 = Formula(formula).spectrum()

    min_fraction: float = 1e-16

    # Create a new dictionary to store the overlapped spectrum
    spectrum = {}
    for key1, items in sorted(spectrum1.items()):
        f = items.fraction
        m = items.mass
        k = items.massnumber
        if key1 in spectrum2:
            f2 = spectrum2[key1].fraction
            m2 = spectrum2[key1].mass
            # Calculate the new mass and relative intensity for the isotope
            m_new = (f * m + f2 * m2 * ratio) / (f + f2 * ratio)
            f_new = f + f2 * ratio
            spectrum[k] = [m_new, f_new, 1.0]
        else:
            # If the isotope is only present in spectrum 1, retain its original mass and relative intensity
            spectrum[k] = [m, f, 1.0]
    # Add isotopes that are exclusively present in spectrum2 to the overlapped spectrum
    for key2, items in sorted(spectrum2.items()):
        if key2 not in spectrum1:
            f = items.fraction
            m = items.mass
            k = items.massnumber
            spectrum[k] = [m, f, 1.0]

    # Filter out isotopes with low intensities and normalize the remaining isotopic intensities to 100
    if min_intensity is not None:
        norm = 100 / max(v[1] for v in spectrum.values())
        for massnumber, value in spectrum.copy().items():
            if value[1] * norm < min_intensity:
                del spectrum[massnumber]
            else:
                spectrum[massnumber][2] = value[1] * norm

    return Spectrum(spectrum)

if __name__ == "__main__":
    spectrum_ = get_dehydroisomer_isotopes("C19H37NO5", 0.33, 0.0001)
    print(spectrum_)
 
    
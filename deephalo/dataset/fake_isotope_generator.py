import numpy as np
from molmass import Formula, Spectrum

class FakeIsotopeGenerator:
    def __init__(self, formula, type=None, rate=None):
        """
        Initialize the FakeIsotopeGenerator class.

        Parameters:
        formula: str, the chemical formula.
        type: str, the type of modification ('hydro', 'Fe', 'B', 'Se', or None).
        rate: float, the rate for hydrogenation (only used for 'hydro' type).
        """
        self.formula = formula
        self.type = type
        self.rate = rate

    def generate(self) -> dict:
        """
        Simulate the isotopic distribution based on the formula and type.

        Returns:
        dict: Simulated isotopic distribution.
        """
        fm = Formula(self.formula)
        if self.type == 'hydro':
            fm_isos = self.get_dehydroisomer_isotopes(self.rate, 0.0001).dataframe()
        elif self.type == 'Fe':
            fm_isos = self.get_iron_additive_isotopes().dataframe()
        elif self.type == 'B':
            fm_isos = self.get_boron_additive_isotopes().dataframe()
        elif self.type == 'Se':
            fm_isos = self.get_selenium_additive_isotopes().dataframe()
        else:
            fm_isos = fm.spectrum(min_intensity=0.000001).dataframe()

        # Get relative mass and intensity
        relative_mass = fm_isos['Relative mass'].tolist()[:7]
        intensity = fm_isos['Intensity %'].tolist()[:7]

        # Pad relative_mass and intensity if their length is less than 7
        while len(relative_mass) < 6:
            relative_mass.append(0)
            intensity.append(0)

        intensity = np.array(intensity) / max(intensity)
        # Convert to dictionary
        dict_isos = {
            'mz_0': relative_mass[0], 'mz_1': relative_mass[1], 'mz_2': relative_mass[2], 'mz_3': relative_mass[3],
            'mz_4': relative_mass[4], 'mz_5': relative_mass[5],
            'p0_int': intensity[0], 'p1_int': intensity[1], 'p2_int': intensity[2], 'p3_int': intensity[3],
            'p4_int': intensity[4], 'p5_int': intensity[5]
        }
        return dict_isos

    def get_dehydroisomer_isotopes(self, ratio, min_intensity=0.0001) -> Spectrum:
        """
        Calculate the overlapped spectrum of a chemical formula and its dehydroisomer.

        Parameters:
        ratio: float, the weighting between the two spectra in the final overlapped spectrum.
        min_intensity: float, the minimum relative intensity for an isotope to be included in the final spectrum.

        Returns:
        Spectrum: the overlapped spectrum.
        """
        spectrum1 = (Formula(self.formula) - Formula("H2")).spectrum()
        spectrum2 = Formula(self.formula).spectrum()

        spectrum = {}
        for key1, items in sorted(spectrum1.items()):
            f = items.fraction
            m = items.mass
            k = items.massnumber
            if key1 in spectrum2:
                f2 = spectrum2[key1].fraction
                m2 = spectrum2[key1].mass
                m_new = (f * m + f2 * m2 * ratio) / (f + f2 * ratio)
                f_new = f + f2 * ratio
                spectrum[k] = [m_new, f_new, 1.0]
            else:
                spectrum[k] = [m, f, 1.0]

        for key2, items in sorted(spectrum2.items()):
            if key2 not in spectrum1:
                f = items.fraction
                m = items.mass
                k = items.massnumber
                spectrum[k] = [m, f, 1.0]

        if min_intensity is not None:
            norm = 100 / max(v[1] for v in spectrum.values())
            for massnumber, value in spectrum.copy().items():
                if value[1] * norm < min_intensity:
                    del spectrum[massnumber]
                else:
                    spectrum[massnumber][2] = value[1] * norm

        return Spectrum(spectrum)

    def get_iron_additive_isotopes(self) -> Spectrum:
        """Get the isotopic distribution of the formula with iron additive."""
        f = Formula(self.formula + "Fe") - Formula('H3')
        return f.spectrum()

    def get_boron_additive_isotopes(self) -> Spectrum:
        """Get the isotopic distribution of the formula with boron additive."""
        f = Formula(self.formula + "B") - Formula('H3')
        return f.spectrum()

    def get_selenium_additive_isotopes(self) -> Spectrum:
        """Get the isotopic distribution of the formula with selenium additive."""
        f = Formula(self.formula + "Se")
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

import numpy as np
from molmass import Formula,Spectrum


def adding_noise_to_intensity (Intensity,sigma_IR=0.05, sigma_IA=0.00005):

    """
    The parameters and function come from the reference:

    'Meusel, M.;  Hufsky, F.;  Panter, F.;  Krug, D.;  Müller, R.; Böcker, S., 
    Predicting the Presence of Uncommon Elements in Unknown Biomolecules from Isotope Patterns.
    Anal Chem 2016, 88 (15), 7556-66.'

    parameters:

    Intesntiy: simulated intensity for a compound by molmass

    sigma_IR = 0.05 for training set, 0.04 and 0.07 for evaluation set
    sigma_IA =0.005 for training set, 0.0015 and 0.006 for evaluation set

    """
    # Generate the relative noise for intensity
    relative_noise = np.random.uniform(1,1+sigma_IR)

    # Generate the absolute noise for intensity
    absolute_noise = np.random.uniform(0, sigma_IA)

    # Calculate the simulated intensity
    I_simulated = Intensity * relative_noise + absolute_noise

    return I_simulated

def adding_noise_to_mass (mass, M=0.0015):

    """
    The parameters and function come from the reference:

    'Meusel, M.;  Hufsky, F.;  Panter, F.;  Krug, D.;  Müller, R.; Böcker, S., 
    Predicting the Presence of Uncommon Elements in Unknown Biomolecules from Isotope Patterns.
    Anal Chem 2016, 88 (15), 7556-66.'

    parameters:

    mass     : simulated mass for a compound by molmass

    M=0.0015 for training set, 0.0013 and 0.0018 for evaluation set

    """
    # Generate the relative noise for mass
    mass_noise = np.random.uniform(-M, M)

    # Calculate the simulated mass
    m_simulated = mass + mass_noise
    
    return m_simulated

def get_iron_additive_isotopes(formula):

    f=Formula(formula+"Fe")-Formula('H3')
    return f.spectrum()

def other_requirements_trainable_clf(formula):
    #将formula列中的公式转为字典
    f_dict = Formula(formula).composition().dataframe().to_dict()['Count']
    #如果f_dict中的keys是[C,H,O,N,S]的子集，则返回1，否则返回0
    if set(f_dict.keys()).issubset(set(['C','H','O','N','S'])):
        return 1
    else:
        return 0

def mass_spectrum_calc(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3):
    #校正质谱数据   
    if a1_mz>0.1:
        a1_a0 = a1_mz - a0_mz
    else:
        a1_a0 = 0
    if a2_mz>0.1:
        a2_a0 = a2_mz - a0_mz
        a2_a1 = a2_mz - a1_mz
    else:
        a2_a0 = 0
        a2_a1 = 0
    
    if a3_mz>0.1:
        a3_a0 = a3_mz - a0_mz
        a3_a1 = a3_mz - a1_mz
        a3_a2 = a3_mz - a2_mz
    else:
        a3_a0 = 0
        a3_a1 = 0
        a3_a2 = 0
    
    if b_1_mz<0.1:
        a0_b1 = 0
    else:
        a0_b1 = a0_mz-b_1_mz

    if b_2_mz<0.1:
        b1_b2 = 0
    else:
        b1_b2 = b_1_mz-b_2_mz
    
    a0_norm = a0/2000
    return a0_norm,a1_a0,a2_a0,a2_a1,a3_a0,a3_a1,a3_a2,a0_b1,b1_b2

def mass_spectrum_calc_2(b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz,b_2,b_1,a0,a1,a2,a3):
    #将b_2_mz,b_1_mz,a0_mz,a1_mz,a2_mz,a3_mz中第一个不为0的值赋给new_a0_mz，其后的值赋给new_a1_mz,new_a2_mz,new_a3_mz

    if b_2_mz != 0:
        new_a0_mz = b_2_mz
        new_a1_mz = b_1_mz
        new_a2_mz = a0_mz
        new_a3_mz = a1_mz
        new_a0_ints = b_2
        new_a1_ints = b_1
        new_a2_ints = 1
        new_a3_ints = a1
    elif b_1_mz != 0:
        new_a0_mz = b_1_mz
        new_a1_mz = a0_mz
        new_a2_mz = a1_mz
        new_a3_mz = a2_mz
        new_a0_ints = b_1
        new_a1_ints = 1
        new_a2_ints = a1
        new_a3_ints = a2

    elif a0_mz != 0:
        new_a0_mz = a0_mz
        new_a1_mz = a1_mz
        new_a2_mz = a2_mz
        new_a3_mz = a3_mz
        new_a0_ints = 1
        new_a1_ints = a1
        new_a2_ints = a2
        new_a3_ints = a3
        
    new_a2_a1 = new_a2_mz - new_a1_mz
    new_a2_a0 = new_a2_mz - new_a0_mz
    new_a2_a0_10 = new_a2_a0**10
    return new_a0_mz,new_a1_mz,new_a2_mz,new_a3_mz,new_a0_ints,new_a1_ints,new_a2_ints,new_a3_ints,new_a2_a1,new_a2_a0,new_a2_a0_10

def get_hydroisomer_isotopes(formula,ratio,min_intensity=0.0001):

    spectrum1=Formula(formula).spectrum()
    spectrum2=Formula(formula+"H2").spectrum()
    # print(spectrum1,spectrum2)

    min_fraction: float = 1e-16
    # min_intensity: float =1e-16
    #新建一个spectrum类

    spectrum={}
    for key1,items in sorted(spectrum1.items()):
        # print(key1)
        f=items.fraction
        m=items.mass
        k=items.massnumber
        if f<min_fraction:
            continue
        if key1 in spectrum2:
            f2=spectrum2[key1].fraction
            m2=spectrum2[key1].mass
            if f2<min_fraction:
                continue
            s_0 = (f * m + f2 * m2*ratio) / (f + f2*ratio)
            s_1 =f + f2*ratio
            spectrum[k]=[s_0, s_1,1.0]
        else:
            spectrum[k] = [m, f,1.0]
    # print(spectrum)
    # return spectrum
    # filter low intensities
    if min_intensity is not None:
        norm = 100 / max(v[1] for v in spectrum.values())
        for massnumber, value in spectrum.copy().items():
            if value[1] * norm < min_intensity:
                del spectrum[massnumber]
            else:
                spectrum[massnumber][2] = value[1]*norm
    # print(spectrum)
    return Spectrum(spectrum)

def get_dehydroisomer_isotopes(formula,ratio,min_intensity=0.0001):

    spectrum2=Formula(formula).spectrum()
    dehydroisomer=Formula(formula)-Formula("H2")
    spectrum1=Formula(str(dehydroisomer)).spectrum()
    # print(spectrum1,spectrum2)

    min_fraction: float = 1e-16
    # min_intensity: float =1e-16

    spectrum={}
    for key1,items in sorted(spectrum1.items()):
        # print(key1)
        f=items.fraction
        m=items.mass
        k=items.massnumber
        if f<min_fraction:
            continue
        if key1 in spectrum2:
            f2=spectrum2[key1].fraction
            m2=spectrum2[key1].mass
            if f2<min_fraction:
                continue
            s_0 = (f * m + f2 * m2*ratio) / (f + f2*ratio)
            s_1 =f + f2*ratio
            spectrum[k]=[s_0, s_1,1.0]
        else:
            spectrum[k] = [m, f,1.0]
    for key2 in spectrum2:
        if key2 not in spectrum1:
            spectrum[key2]=[spectrum2[key2].mass,spectrum2[key2].fraction,1.0]


    # print(spectrum)
    # return spectrum
    # filter low intensities
    if min_intensity is not None:
        norm = 100 / max(v[1] for v in spectrum.values())
        for massnumber, value in spectrum.copy().items():
            if value[1] * norm < min_intensity:
                del spectrum[massnumber]
            else:
                spectrum[massnumber][2] = value[1]*norm
    return Spectrum(spectrum)
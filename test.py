from pyteomics import mzxml

# file = r'D:\python\wangmengyuan\halo_standard_pure\halo_standard_pure\mzXML\4_Br.mzXML'
file = r'H:\opendatabase\massive_dataset\MSV000082428\mzXML_files\5_ACN_1_1.mzXML'
spectra =mzxml.read(file)
for spectrum in spectra:
    if spectrum['msLevel'] == 2:
        print(spectrum) 
        break  
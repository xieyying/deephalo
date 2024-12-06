import pyopenms
import os

# Load the mzML file
exp = pyopenms.MSExperiment()
folder = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\Simulated_LC_MS\LC_MS_information_from_papers\simulated_mzml\mzml'
files = os.listdir(folder)
for file in files:
    if file.endswith('.mzML'):
        try:
            pyopenms.MzMLFile().load(os.path.join(folder, file), exp)
            print(f"mzML file {file} loaded successfully")
        except Exception as e:
            print(f"{file} Error loading mzML file: {e}")
        

import pandas as pd
import os
import subprocess
from pyteomics import mzml
from .methods import judge_charge
from .methods import feature_extractor,correct_df_charge
import tensorflow as tf

class mzml_base:
    def __init__(self) -> None:
        pass

    def load_mzml_file(path,level=1):
        spectra = mzml.read(path,use_index=True,read_schema=True)
        level_spectra = [s for s in spectra if s.get('ms level') == level]
        return level_spectra
    
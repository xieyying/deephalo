import pandas as pd
from molmass import Formula
from multiprocessing import Pool
from .methods_main import create_data
from functools import partial
import os

class Dataset():
    def __init__(self, path, key) -> None:
        """
        This class can construct a basic dataset based on the formula of real compounds in the database.
        It can also construct datasets with noise, simulate chelated iron datasets, simulate dehydrogenation datasets, and simulate hydrogenation datasets.

        Parameters:
        path: str, file path of the real database
        key: str, column name of the formula in the real database

        Attributes:
        data: pd.DataFrame, formulas of real compounds in the database

        Methods:
        filter: Filter data
        filt: Map filter using multiprocessing, default uses all CPUs
        create_dataset: Construct a specified dataset based on the formulas in self.data
        save: Save the dataset as a CSV file
        work_flow: Workflow to create a specified dataset

        Example usage:
        test = Dataset('test.json', 'mol_formula')
        test.work_flow(100, 1000, ['C', 'H', 'O', 'N', 'S'], 'hydro')
        """
        # Read the formula of real compounds from the database
        if path.endswith('.json'):
            self.data = pd.read_json(path)[[key]].dropna()
            # Rename the key column to 'formula'
            self.data = self.data.rename(columns={key: 'formula'})
        elif path.endswith('.csv'):
            self.data = pd.read_csv(path, low_memory=False)[[key]].dropna()
            self.data = self.data.rename(columns={key: 'formula'})

        # Remove duplicates in self.data
        self.data = self.data.drop_duplicates(subset=['formula'], keep='first')

        # Reset the index
        self.data = self.data.reset_index(drop=True)
        
    def filter(self, Multi_core_run_parameters):
        """
        Filter formulas based on mass range and target elements.

        Parameters:
        Multi_core_run_parameters: tuple, contains index, mass range, and target elements

        Returns:
        df: pd.DataFrame, filtered formulas
        """
        df = pd.DataFrame(columns=['formula'])
        try:
            i, mz_start, mz_end, elements_list = Multi_core_run_parameters
            formula = self.data.iloc[i]['formula']
            formula_mass = Formula(formula).isotope.mass
            formula_dict = Formula(formula).composition().dataframe().to_dict()['Count']
            formula_dict_keys = list(formula_dict.keys())
            if formula_mass >= mz_start and formula_mass <= mz_end:
                # If all elements in formula_dict_keys are in elements_list, add the formula to df
                if set(formula_dict_keys).issubset(set(elements_list)):
                    df = pd.concat([df, pd.DataFrame([[formula]], columns=['formula'])], ignore_index=True)
            return df
        except:
            return df
        
    def filt(self, mz_start, mz_end, elements_list):
        """
        Map filter using multiprocessing, default uses all CPUs.

        Parameters:
        mz_start: int, minimum mass
        mz_end: int, maximum mass
        elements_list: list, target elements

        Returns:
        Modifies self.data directly, no return value.
        """
        pool = Pool(4)
        dfs = pool.map(self.filter, [(i, mz_start, mz_end, elements_list) for i in range(len(self.data))])
        pool.close()
        df = pd.concat(dfs, ignore_index=True)

        self.data = df

    def create_dataset(self, type, rates):  # return_from_max_ints):
        """
        Construct a specific dataset based on the formulas in self.data. Different types correspond to different datasets:

        Basic dataset: Based on the formulas of real compounds in the database.
        Noise-added dataset: Based on the basic dataset with added noise.
        Simulated chelated iron dataset: Simulated data after chelation with iron.
        Simulated dehydrogenation dataset: Simulated data after dehydrogenation.
        Simulated hydrogenation dataset: Simulated data after hydrogenation.

        Parameters:
        type: str, dataset type, options are 'base', 'Fe', 'B', 'Se', 'hydro'.
        rates: list, hydrogenation rates.

        """
        if type in ['base', 'Fe', 'B', 'Se', '2M', '2M-Cl-Br']:
            # Basic dataset
            pool = Pool(4)
            func = partial(create_data, type=type)  # return_from_max_ints=return_from_max_ints)
            dfs = pool.map(func, [formula for formula in self.data['formula']])
            pool.close()
            df = pd.concat(dfs, ignore_index=True)
            self.df_data = df
            
        elif type == 'hydro':
            df = pd.DataFrame()
            # Simulated hydrogenation dataset
            pool = Pool(4)
            for rate in rates:
                func = partial(create_data, type=type, rate=rate)  # return_from_max_ints=return_from_max_ints)
                dfs = pool.map(func, [formula for formula in self.data['formula']])
                df0 = pd.concat(dfs, ignore_index=True)
                df = pd.concat([df, df0], ignore_index=True)
                print('done rate:', rate)
                print('temp data:', len(df0))
                print('total data:', len(df))
            pool.close()            
            self.df_data = df.sample(frac=(1 / (len(rates))), random_state=42)
            
        else:
            raise ValueError('type must be in [base, Fe, B, Se, hydro, 2M, 2M-Cl-Br]')
        
    def save(self, path):
        """
        Save the dataset as a CSV file.

        Parameters:
        path: str, save path
        """
        # Filter, keep only data with group <= 7
        self.df_data = self.df_data[self.df_data['group'] <= 7]
        self.df_data = self.df_data[self.df_data['mz_0'] <= 2000]
        self.df_data.to_csv(path, index=False)

    def work_flow(self, para, type):
        """
        Workflow to create a specified dataset.

        Parameters:
        para: object, contains parameters like min_mass, max_mass, target_elements, etc.
        type: str, dataset type, options are 'base', 'Fe', 'B', 'Se', 'hydro'.

        Returns:
        This process takes a long time, so the dataset is saved as a file.
        """
        self.filt(para.mz_start, para.mz_end, para.elements_list)
        self.create_dataset(type, para.rate_for_hydro)  # para.return_from_max_ints)
        
        # Create the dataset folder if it doesn't exist
        if not os.path.exists('./dataset'):
            os.mkdir('./dataset')
        self.save('./dataset/' + type + '.csv')

class Datasets(Dataset):
    """
    Subclass of Dataset, used to merge multiple datasets.

    Parameters:
    datas: list, list of datasets
    """
    def __init__(self, datas) -> None:
        self.data = pd.concat(datas, axis=0)
        print('Original data:', len(self.data))
        # Remove duplicates in self.data
        self.data = self.data.drop_duplicates(subset=['formula'], keep='first')
        print('Data after deduplication:', len(self.data))
        # Reset the index
        self.data = self.data.reset_index(drop=True)
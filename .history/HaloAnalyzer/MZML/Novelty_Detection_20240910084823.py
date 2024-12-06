import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

class NoveltyDetection:
    def __init__(self, features, train_path, test_path) -> None:
        train = pd.read_csv(train_path)
        train['normalized_mz'] = train['mz_0']/2000
        test = pd.read_csv(test_path)
        test['normalized_mz'] = test['mz']*test['charge']/2000
        self.features = features
        # self.train = train[train['group'].isin([0, 1, 2, 6])]
        self.train = train[train['group'].isin([0, 1, 2, 3,4,5, 6])]
        self.test = test

    def model_train(self, model, method_name):
        model.fit(self.train[self.features])
        self.model = model
        self.method_name = method_name

    def model_predict_and_evaluate(self):
        self.predictions = self.model.predict(self.test[self.features])
        print("len of test:",len(self.test))
        print('nums of -1:',np.sum(self.predictions == -1))
        print('detail of -1:',self.test[self.predictions == -1])
        self.test[self.predictions == -1].to_csv(f'C:\\Users\\xyy\\Desktop\\tem\\{self.method_name}_test_0.008.csv',index=False)
        print('model:',self.method_name,'done')
        print('-----------------------------------')

if __name__ == "__main__":
    def method_select(method_name):
        if method_name == 'lof':
            return LocalOutlierFactor(
                algorithm='auto',
                n_neighbors=10,
                novelty=True,
                # contamination=0.001,
                n_jobs=-1
                )
        elif method_name == 'elliptic':
            return EllipticEnvelope(
                contamination=0.01
                )
        elif method_name == 'isolation':
            return IsolationForest(
                n_estimators=200,
                max_samples=1.0,
                contamination=0.008,
                max_features=1.0,
                bootstrap=True,
                verbose=1,
                )

    features = [
            "p0_int",
            "p1_int",
            "p2_int",
            "p3_int",
            "p4_int",
            "p5_int",
            "m2_m1",
            "m1_m0",
            # "normalized_mz",
        ]
    file_path = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\feature_static\base\adding_noise_inty_mz_0.001\base.csv'
    # test_file = r'C:\Users\xq75\Desktop\p_test\result\scan_for_model_input.csv'
    test_file = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\Simulated_LC_MS\LC_MSMS_data_from_papers\simulated_mzml\haloanalyzer_analysis_results\2ppm_2e4_feature.csv'

    new_test = NoveltyDetection(features, file_path, test_file)
    # test_methods = [ 'elliptic', 'isolation']#'lof',
    test_methods = [ 'isolation']
    for method_name in test_methods:
        model = method_select(method_name)
        new_test.model_train(model, method_name)
        new_test.model_predict_and_evaluate()
        


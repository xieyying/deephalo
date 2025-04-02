from collections import Counter



def halo_scoring(df_f_, df_scan_):
    """
    Evaluate the probability of features based on both feature isotope patterns and scan based isotope patterns
    """
    df_f = df_f_.copy()
    df_scan = df_scan_.copy()

    # Create a dictionary to store the results
    results = {
        'feature_id_flatten': [],
        'scan_based_halo_class': [],
        'scan_based_halo_score': [],
        'scan_based_halo_sub_class': [],
        'scan_based_halo_sub_score': [],
        'scan_based_halo_ratio': []
    }

    for i in df_scan['feature_id_flatten'].unique():
        I = df_scan[df_scan['feature_id_flatten'] == i]['Feature_based_prediction'].tolist()
        scan_based_halo_class, scan_based_halo_score, scan_based_halo_sub_class, scan_based_halo_sub_score, scan_based_halo_ratio = roi_scan_based_halo_evaluation(I)
        
        results['feature_id_flatten'].append(i)
        results['scan_based_halo_class'].append(scan_based_halo_class)
        results['scan_based_halo_score'].append(scan_based_halo_score)
        results['scan_based_halo_sub_class'].append(scan_based_halo_sub_class)
        results['scan_based_halo_sub_score'].append(scan_based_halo_sub_score)
        results['scan_based_halo_ratio'].append(scan_based_halo_ratio)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Merge results with df_scan and df_f
    df_scan = df_scan.merge(results_df, on='feature_id_flatten', how='left')
    df_f = df_f.merge(results_df.rename(columns={'feature_id_flatten': 'feature_id'}), on='feature_id', how='left')

    # Calculate feature_based_halo_score
    df_f['feature_based_halo_score'] = df_f['Feature_based_prediction'].apply(lambda x: 1 if x in [0, 1, 2] else 0)

    # Calculate H_score
    df_f['H_score'] = (df_f['scan_based_halo_score']) / 300 + (df_f['scan_based_halo_ratio']) / 3 + (df_f['feature_based_halo_score']) / 3
    df_scan['H_score'] = (df_scan['scan_based_halo_score']) / 300 + (df_scan['scan_based_halo_ratio']) / 3 + (df_scan['Feature_based_prediction'].apply(lambda x: 1 if x in [0, 1, 2] else 0)) / 3

    return df_f, df_scan


def zigzag_score(I):
    """
    Calculate the ZigZag score based on the classification results of all scans in an ROI
    """
    # Calculate the maximum and minimum values of I
    Imax= max(I)
    Imin = min(I)
    N = len(I) 
    total = 0
    # Calculate the ZigZag score for I
    for n in range(1,N-1):
        term = (2 * I[n] - I[n - 1] - I[n + 1])**2 
        total += term
    zigzag = total/(N*(Imax-Imin)**2)
    # Convert the ZigZag score to a percentage
    Zscore = (4-8/N-zigzag)/(4-8/N)*100 * halo_ratio(I)
    return Zscore
def score_scans(I):
    """
    Determine the probability of an ROI being a halo based on the classification results of all scans in the ROI
    """
    # Get the common classes in the ROI
    com_class = list(Counter(I).keys())
    counter = Counter(I)


    # Determine the halo classification for the ROI
    if any(i in com_class for i in [0, 1, 2]):
        scan_based_halo_class = 'halo'
        if len(com_class) == 1:
            if len(I) >= 2:
                scan_based_halo_score = 100
                scan_based_halo_sub_score = 100
                scan_based_halo_sub_class = com_class[0]
            else:
                scan_based_halo_score = 0
                scan_based_halo_sub_class = "None"
                scan_based_halo_sub_score = "None"
            
        else:
            if {0, 1, 2}.issuperset(set(com_class)):
                scan_based_halo_score = 100
                scan_based_halo_sub_class =max(counter.items(), key=lambda x: x[1])[0]
                scan_based_halo_sub_class_ratio = counter[scan_based_halo_sub_class] / len(I)
                if len(I) > 2:
                    scan_based_halo_score = zigzag_score(I)
                else:
                    scan_based_halo_sub_score = halo_ratio(I) 
            else:
                I_new = [1 if i in [0,1,2] else 0 for i in I]
                if len(I) > 2:
                    scan_based_halo_score = zigzag_score(I)
                else:
                    scan_based_halo_score = halo_ratio(I) 
                scan_based_halo_sub_class = "None"
                scan_based_halo_sub_score = "None"
    else:
        scan_based_halo_class = 'non-halo'
        scan_based_halo_score = 0
        scan_based_halo_sub_class = 'None'
        scan_based_halo_sub_score = 'None'

    return scan_based_halo_class,scan_based_halo_score,scan_based_halo_sub_class,scan_based_halo_sub_score

def halo_ratio(I):
    """
    Calculate the ratio of halo features in a given list of features.
    """
    # Count the number of halo features
    halo_count = sum(1 for i in I if i in [0, 1, 2])
    # Calculate the ratio of halo features
    halo_ratio = halo_count / len(I) if len(I) > 0 else 0
    return halo_ratio

def score_feature(group):
    return 1 if group in [0, 1, 2] else 0

def overall_score(I, feature_g):
    """
    Calculate the overall score based on scan and feature scores.
    """
    # Calculate the overall score
    overall_score = 0.3 * score_scans(I) + 0.3 * halo_ratio(I) + 0.4 * score_feature(feature_g)
    return overall_score

class HalogenScorer:
    """
    Class to calculate the H_score based on the classification results of all scans in a feature.
    
    """
    
    def __init__(self,halogen_classes=None):
        """
        Initialize the HalogenScorer class.
        
        Args:
            halogen_classes (set, optional): Set of class indices that represent halogen patterns.
                                           Defaults to {0, 1, 2}.
        """
        self.halogen_classes = halogen_classes or {0, 1, 2}
    
    def calculate_halogen_ratio(self, scan_predictions):
        """
        Calculate the ratio of halogen patterns in the scan predictions.
        
        Args:
            scan_predictions (list): List of scan predictions for a feature.
            
        Returns:
            float: Ratio of halogen patterns in the scan predictions.
        """
        halogen_count = sum(1 for pred in scan_predictions if pred in self.halogen_classes)
        halo_ratio = halogen_count / len(scan_predictions) if len(scan_predictions) > 0 else 0
        return halo_ratio
    
    def calculate_prediction_consistency(self, scan_predictions):
        
        _halo_ratio = self.calculate_halogen_ratio(scan_predictions)
        
        if len(scan_predictions) <= 2:
            return _halo_ratio * 100
        
        max_pred = max(scan_predictions)
        min_pred = min(scan_predictions)
        
        if max_pred == min_pred:
            return _halo_ratio * 100
        
        n_scans = len(scan_predictions)
        total_derviation = 0
        
        # Calculate prediction consistency using zigzag algorithm
        for i in range(1, n_scans - 1):
            deviation = (2 * scan_predictions[i] - scan_predictions[i - 1] - scan_predictions[i + 1]) ** 2
            total_derviation += deviation
        
        # Normalize the deviation
        normalized_deviation = total_derviation / (n_scans * (max_pred - min_pred) ** 2)
        consistency_score = (4 - 8 / n_scans - normalized_deviation) / (4 - 8 / n_scans) * 100
        
        # Weigh the consistency score by the halo ratio
        consistency_score = consistency_score * _halo_ratio
        return consistency_score
    
    def 

    


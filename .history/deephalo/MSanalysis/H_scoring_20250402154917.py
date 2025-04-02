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
        
        if len(scan_predictions) == 2:
            return _halo_ratio * 100
        elif len(scan_predictions) == 1:
            return 0
        
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
    
    def score_scan_predictions(self, scan_predictions):
        """
        Evaluate the scan predictions and return the classification and scores.
        
        Args:
            scan_predictions (list): List of scan predictions for a feature.
            
        Returns:
            tuple: Classification, score, and sub-classification.
        """
    
        if any(pred in self.halogen_classes for pred in scan_predictions):
            classification = 'halo'
            if self.halogen_classes.issubset(set(scan_predictions)):
                score = 100
                sub_classification = max(Counter(scan_predictions).items(), key=lambda x: x[1])[0]
            else:
                _scan_predictions = [1 if pred in self.halogen_classes else 0 for pred in scan_predictions]
                score = self.calculate_prediction_consistency(_scan_predictions)
                if halo_ratio(scan_predictions) > 0.5:
                    scan_predictions_tem = [pred for pred in scan_predictions if pred in self.halogen_classes]
                    sub_classification = max(Counter(scan_predictions_tem).items(), key=lambda x: x[1])[0]
                else:
                    sub_classification = 'None'                  
        else:
            classification = 'non-halo'
            score = 0
            sub_classification = 'None'
        
        return classification, score, sub_classification
    
    def score_feature_prediction(self, feature_prediction):
        """
        Score the feature based on its predictions.
        
        Args:
            feature_predictions (list): List of predictions for a feature.
            
        Returns:
            float: Feature score.
        """
        return 1 if feature_prediction in self.halogen_classes else 0
    
    def calculate_overall_score(self, scan_predictions, feature_prediction):
        """
        Calculate the overall score based on scan and feature scores.
        
        Args:
            scan_predictions (list): List of scan predictions for a feature.
            feature_prediction (int): Feature prediction.
            
        Returns:
            float: Overall score.
        """
        classification,scan_score, scan_sub_class = self.score_scan_predictions(scan_predictions)
        halo_ratio = self.calculate_halogen_ratio(scan_predictions)
        feature_score = self.score_feature_prediction(feature_prediction)
        H_score = (scan_score) / 3 + (halo_ratio) / 3 + (feature_score) / 3
        return classification, H_score, scan_sub_class
     

from collections import Counter

class HalogenScorer:
    """
    A class to evaluate and score the probability of halogen presence in 
    mass spectrometry features based on scan level isotope pattern predictions 
    and feature level isotope pattern predictions.

    This class provides methods to:
    1. Calculate the ratio of halogen patterns in scan predictions
    2. Evaluate pattern consistency across scans using zigzag algorithm
    3. Score individual scans and features for halogen presence
    4. Combine multiple metrics into an overall halogen confidence score (H_score)
    """
        
    def __init__(self, halogen_classes=None):
        """
        Initialize the HalogenScorer.
        
        Args:
            halogen_classes (set, optional): Class indices representing halogen patterns.
                                           Defaults to {0, 1, 2}.
        """
        # Define classes that represent halogen patterns
        self.halogen_classes = halogen_classes or {0, 1, 2}
    
    def calculate_halogen_ratio(self, scan_predictions):
        """
        Calculate the ratio of halogen-positive scans to total scans (R_halo).
        
        Args:
            scan_predictions (list): List of classification results for multiple scans.
            
        Returns:
            float: Ratio of halogen patterns (0.0-1.0).
        """
        # Handle empty list case
        if not scan_predictions:
            return 0
            
        # Count scans classified as halogen
        halogen_count = sum(1 for pred in scan_predictions if pred in self.halogen_classes)
        # Calculate ratio
        r_halo = halogen_count / len(scan_predictions)
        return r_halo
    
    def calculate_prediction_consistency(self, scan_predictions):
        """
        Calculate how consistent the predictions are across scans using the zigzag algorithm.

        Higher scores indicate more consistent patterns. The score is weighted by the 
        proportion of halogen-positive scans.

        Args:
            scan_predictions (list): List of element prediction classifications of scan level isotope patterns for a feature.
            
        Returns:
            float: Consistency score (0-100).
        """

        # Get halogen ratio for weighting
        halogen_ratio = self.calculate_halogen_ratio(scan_predictions)
        
        # Handle edge cases
        if len(scan_predictions) <= 1:
            return 0
        elif len(scan_predictions) == 2:
            return halogen_ratio * 100
            
        # If all predictions are the same, return perfect consistency
        max_pred = max(scan_predictions)
        min_pred = min(scan_predictions)
        if max_pred == min_pred:
            return halogen_ratio * 100
        
        # Calculate consistency using zigzag algorithm
        n_scans = len(scan_predictions)
        total_deviation = 0
        
        for i in range(1, n_scans - 1):
            # Compare each point with its neighbors
            # A perfect pattern would have points following a consistent trend
            deviation = (2 * scan_predictions[i] - scan_predictions[i - 1] - scan_predictions[i + 1]) ** 2
            total_deviation += deviation
        
        # Normalize the deviation based on number of scans and prediction range
        try:
            normalized_deviation = total_deviation / (n_scans * (max_pred - min_pred) ** 2)
            # Convert to a score from 0-100 (higher is better)
            base_score = (4 - 8/n_scans - normalized_deviation) / (4 - 8/n_scans) * 100
            # Ensure score is within valid range
            base_score = base_score
        except (ZeroDivisionError, ValueError):
            # Handle possible division by zero or other errors
            base_score = 0
        
        # Weight the consistency score by the halogen ratio
        weighted_score = base_score * halogen_ratio
        return weighted_score
    
    def score_scan_predictions(self, scan_predictions):
        """
        Evaluate scan predictions to determine halogen presence and pattern quality.
        
        Args:
            scan_predictions (list): List of classification results for scans.
            
        Returns:
            tuple: (classification, score, sub_classification)
                classification: 'halo' or 'non-halo'
                score: Confidence score (0-100)
                sub_classification: Most common halogen class or 'None'
        """
        # Handle empty input
        if not scan_predictions:
            return 'non-halo', 0, 'None'
        
        # Check if any halogen class is present
        if any(pred in self.halogen_classes for pred in scan_predictions):
            classification = 'halo'
            
            # Check if all predictions are among the defined halogen classes
            if self.halogen_classes.issuperset(set(scan_predictions)):
                # Perfect halogen pattern
                score = 100
                # Find the most common halogen class
                sub_classification = max(Counter(scan_predictions).items(), key=lambda x: x[1])[0]
            else:
                # Mixed pattern with some non-halogen classes
                # Convert to binary (halogen vs non-halogen)
                binary_predictions = [1 if pred in self.halogen_classes else 0 for pred in scan_predictions]
                # Calculate consistency score based on binary pattern
                score = self.calculate_prediction_consistency(binary_predictions)
                
                # Determine sub-classification
                # If majority are halogen classes, find most common
                if self.calculate_halogen_ratio(scan_predictions) > 0.5:
                    # Filter to only include halogen classes
                    halogen_predictions = [pred for pred in scan_predictions if pred in self.halogen_classes]
                    # Find most common halogen class
                    sub_classification = max(Counter(halogen_predictions).items(), key=lambda x: x[1])[0]
                else:
                    sub_classification = 'None'
        else:
            # No halogen classes detected
            classification = 'non-halo'
            score = 0
            sub_classification = 'None'
        
        return classification, score, sub_classification
    
    def score_feature_prediction(self, feature_prediction):
        """
        Score a single feature prediction based on halogen presence.
        
        Args:
            feature_prediction: The element prediction classification based on the feature level isotope pattern for a feature.
            
        Returns:
            int: 1 if feature belongs to a halogen class, 0 otherwise.
        """
        return 1 if feature_prediction in self.halogen_classes else 0
    
    def calculate_overall_score(self, scan_predictions, feature_prediction):
        """
        Calculate a comprehensive halogen score combining multiple metrics.
        
        This H_score combines:
        1. Scan-based pattern consistency
        2. Proportion of halogen-positive scans
        3. Feature-level halogen prediction
        
        Args:
            scan_predictions (list): List of scan predictions for a feature.
            feature_prediction: The feature's classification value.
            
        Returns:
            tuple: (classification, H_score, sub_classification)
                classification: 'halo' or 'non-halo'
                H_score: Overall confidence score (0-1)
                sub_classification: Most common halogen class or 'None'
        """
        # Get scan-based scores and classification
        classification, scan_score, sub_classification = self.score_scan_predictions(scan_predictions)
        
        # Calculate halogen ratio (proportion of halogen-positive scans)
        halogen_ratio = self.calculate_halogen_ratio(scan_predictions)
        
        # Get binary feature score
        feature_score = self.score_feature_prediction(feature_prediction)
        
        # Calculate overall score as weighted average of components
        # Scan score is divided by 3 (not 300) for consistent weighting
        h_score = (scan_score / 3) + (halogen_ratio / 3) + (feature_score / 3)
        
        return classification, h_score, sub_classification
    
    def process_feature(self, scan_predictions, feature_prediction):
        """
        Comprehensive processing of a single feature.
        
        Args:
            scan_predictions (list): List of scan predictions for the feature.
            feature_prediction: The feature's classification value.
            
        Returns:
            dict: Dictionary with all scoring results.
        """
        # Calculate basic metrics
        halogen_ratio = self.calculate_halogen_ratio(scan_predictions)
        
        # Get overall classification and score
        classification, h_score, sub_classification = self.calculate_overall_score(
            scan_predictions, feature_prediction
        )
        
        # Get consistency score
        if len(scan_predictions) > 1:
            consistency_score = self.calculate_prediction_consistency(scan_predictions)
        else:
            consistency_score = 0
            
        # Feature-level score
        feature_score = self.score_feature_prediction(feature_prediction)
        
        # Return comprehensive results
        return {
            'classification': classification,
            'h_score': h_score,
            'sub_classification': sub_classification,
            'halogen_ratio': halogen_ratio,
            'consistency_score': consistency_score,
            'feature_score': feature_score
        }
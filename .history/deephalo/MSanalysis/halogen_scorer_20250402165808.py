from collections import Counter
import pandas as pd

class HalogenScorer:
    """
    A class to evaluate and score the probability of halogen presence in 
    a mass spectrometry feature based on scan level isotope pattern predictions (scan predictions) 
    and the feature level isotope pattern prediction.

    This class provides methods to:
    1. Calculate the ratio of halogen-positive scans to total scans (R_halo)
    2. Evaluate pattern consistency across scans using zigzag algorithm (Z_score)
    3. Score the feature for halogen presence (F_score)
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
            scan_predictions (list): List of element prediction classifications of 
                                    scan level isotope patterns for the feature.
            
        Returns:
            float: Ratio of halogen-positive scans (0.0-1.0).
        """
        # Handle empty list case
        if not scan_predictions:
            return 0
            
        # Count scans classified as halogen
        halogen_count = sum(1 for pred in scan_predictions if pred in self.halogen_classes)
        # Calculate ratio
        r_halo = halogen_count / len(scan_predictions)
        return r_halo
    
    def calculate_zigzag_score(self, scan_predictions):
        """
        Calculate how consistent the predictions are across scans using the zigzag algorithm (Z_score).
        
        Higher scores indicate more consistent patterns. The score measures the smoothness of
        transitions between adjacent scan predictions and is weighted by the proportion of
        halogen-positive scans.

        Args:
            scan_predictions (list): List of element prediction classifications of 
                                    scan level isotope patterns for a feature.
            
        Returns:
            float: ZigZag consistency score (0.00-1.00).
        """
        # Get halogen ratio for weighting the final score
        halogen_ratio = self.calculate_halogen_ratio(scan_predictions)
        
        # Convert to binary representation (halogen vs non-halogen)
        binary_predictions = [1 if pred in self.halogen_classes else 0 for pred in scan_predictions]
        
        # Handle edge cases
        if len(binary_predictions) <= 1:
            return 0
        elif len(binary_predictions) == 2:
            return halogen_ratio
            
        # If all predictions are the same, return perfect consistency weighted by halogen ratio
        max_pred = max(binary_predictions)
        min_pred = min(binary_predictions)
        if max_pred == min_pred:
            return halogen_ratio
        
        # Calculate consistency using zigzag algorithm
        n_scans = len(binary_predictions)
        total_deviation = 0
        
        for i in range(1, n_scans - 1):
            # Calculate deviation for each point compared to its neighbors
            # The formula measures how much each point deviates from the expected linear trend
            deviation = (2 * binary_predictions[i] - binary_predictions[i - 1] - binary_predictions[i + 1]) ** 2
            total_deviation += deviation
        
        try:
            # Normalize the deviation based on number of scans and prediction range
            normalized_deviation = total_deviation / (n_scans * (max_pred - min_pred) ** 2)
            
            # Convert to a score from 0-1 (higher is better)
            base_score = (4 - 8/n_scans - normalized_deviation) / (4 - 8/n_scans)
            
            # Ensure score is within valid range
            base_score = base_score
        except (ZeroDivisionError, ValueError):
            # Handle possible division by zero or other errors
            base_score = 0
        
        # Weight the consistency score by the halogen ratio from 0-1 (higher is better)
        weighted_score = base_score * halogen_ratio
        return weighted_score
    
    def identify_halogen_subclass(self, scan_predictions):
        """
        Identify the most likely halogen subclass based on scan predictions.
        
        This method determines the dominant halogen pattern type among the scans.
        The numeric class values represent specific halogen patterns:
        
        0: Complex halogen patterns - Clₙ/Brₘ (n > 3, m > 1, or both Cl and Br)
        1: Medium complexity - Br/Cl₃
        2: Simple pattern - Cl/Cl₂
        
        Args:
            scan_predictions (list): List of element prediction classifications of
                                    scan level isotope patterns for a feature.
            
        Returns:
            int or str: Most common halogen class if present, otherwise 'None'.
        """
        # Handle empty input
        if not scan_predictions:
            return 'None'

        # Check if any halogen class is present
        if any(pred in self.halogen_classes for pred in scan_predictions):
            # If all predictions are halogen classes
            if self.halogen_classes.issuperset(set(scan_predictions)):
                # Find the most common halogen class
                return max(Counter(scan_predictions).items(), key=lambda x: x[1])[0]
            
            # Mixed pattern with some non-halogen classes
            # If majority are halogen classes, find most common among them
            if self.calculate_halogen_ratio(scan_predictions) > 0.5:
                # Filter to only include halogen classes
                halogen_predictions = [pred for pred in scan_predictions if pred in self.halogen_classes]
                if halogen_predictions:  # Ensure we have halogen predictions
                    return max(Counter(halogen_predictions).items(), key=lambda x: x[1])[0]
        
        # No clear halogen subclass identified
        return 'None'
    
    def score_feature_prediction(self, feature_prediction):
        """
        Score a single feature prediction based on halogen presence (F_score).
        
        Args:
            feature_prediction: The element prediction classification based on 
                               the feature level isotope pattern.
            
        Returns:
            int: 1 if feature belongs to a halogen class, 0 otherwise.
        """
        return 1 if feature_prediction in self.halogen_classes else 0
    
    def calculate_overall_score(self, scan_predictions, feature_prediction):
        """
        Calculate a comprehensive halogen score (H_score) combining multiple metrics.
        
        This H_score combines:
        1. Scan-based pattern consistency (Z_score)
        2. Proportion of halogen-positive scans (R_halo)
        3. Feature-level halogen prediction (F_score)
        
        Args:
            scan_predictions (list): List of element prediction classifications of
                                    scan level isotope patterns for a feature.
            feature_prediction: The feature-level element prediction classification.
            
        Returns:
            tuple: (h_score, sub_class, r_halo, z_score, f_score)
                h_score: Overall halogen confidence score (0-1)
                sub_class: Most likely halogen subclass or 'None'
                r_halo: Ratio of halogen patterns (0-1)
                z_score: Pattern consistency score (0-1)
                f_score: Feature-level score (0 or 1)
        """

        # Calculate component scores
        r_halo = self.calculate_halogen_ratio(scan_predictions)
        z_score = self.calculate_zigzag_score(scan_predictions)
        f_score = self.score_feature_prediction(feature_prediction)
        
        # Calculate weighted H_score
        # Weights: 30% for Z_score, 30% for R_halo, 40% for F_score
        h_score = z_score/3 + r_halo/3 + f_score/3
        
        # Identify most likely halogen subclass
        sub_class = self.identify_halogen_subclass(scan_predictions)
        
        return h_score, sub_class,r_halo, z_score, f_score
    
    def process_feature(self, scan_predictions, feature_prediction):
        """
        Perform comprehensive analysis of a single feature.
        
        This method computes all relevant metrics for a feature and returns
        them in a structured format.
        
        Args:
            scan_predictions (list): List of element prediction classifications of
                                    scan level isotope patterns for a feature.
            feature_prediction: The feature-level element prediction classification.
            
        Returns:
            dict: Complete analysis results including:
                - h_score: Overall confidence score
                - sub_class: Most likely halogen subclass
                - r_halo: Ratio of halogen patterns
                - z_score: Pattern consistency score
                - f_score: Feature-level score
        """
  
        # Calculate overall score and identify subclass
        h_score, sub_class,r_halo, z_score, f_score = self.calculate_overall_score(scan_predictions, feature_prediction)
        
        # Return comprehensive results
        return {
            'h_score': h_score,
            'sub_class': sub_class,
            'r_halo': r_halo,
            'z_score': z_score,
            'f_score': f_score
        }


def halo_scoring(df_f_, df_scan_):
    """
    Evaluate the probability of features based on both feature isotope patterns and scan based isotope patterns
    """
    df_f = df_f_.copy()
    df_scan = df_scan_.copy()
  
    # Create a dictionary to store the results
    results = {
        'feature_id_flatten': [],
        'h_score': [],
        'sub_class': [],
        'r_halo': [],
        'z_score': [],
        'f_score': []
    }

    for i in df_scan['feature_id_flatten'].unique():
        scan_predictions = df_scan[df_scan['feature_id_flatten'] == i]['classification'].tolist()
        feature_prediction = df_f[df_f['feature_id'] == i]['classification'].value()
        h_score, sub_class, r_halo, z_score, f_score = HalogenScorer().process_feature(scan_predictions, feature_prediction)

        
        results['feature_id_flatten'].append(i)
        results['h_score'].append(h_score)
        results['sub_class'].append(sub_class)
        results['r_halo'].append(r_halo)
        results['z_score'].append(z_score)
        results['f_score'].append(f_score)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Merge results with df_scan and df_f
    df_scan = df_scan.merge(results_df, on='feature_id_flatten', how='left')
    df_f = df_f.merge(results_df.rename(columns={'feature_id_flatten': 'feature_id'}), on='feature_id', how='left')

    return df_f, df_scan

if __name__ == "__main__":
    import os
    work_folder = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms_noClFe\2M_fake_molecules\11-23_739\result\test'
    
    df_f_ = pd.read(os.path.join(work_folder, 'feature_predictions.csv'))
    df_scan_ = pd.read(os.path.join(work_folder, 'scan_predictions.csv'))

    
    
    
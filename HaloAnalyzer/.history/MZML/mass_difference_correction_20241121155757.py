import numpy as np
import pandas as pd


def feature_classifier(masstrace_centroid_mz: list,charge:int) -> int:
    """
    Classify features based on the mass difference between the centroid m/z values of the masstrace.

    Args:
        masstrace_centroid_mz (list): List of centroid m/z values.

    Returns:
        int: Classification label (0, 1, 2, 3) or None if no classification criteria are met.
    """
    if not isinstance(masstrace_centroid_mz, list):
        raise TypeError("masstrace_centroid_mz must be a list of m/z values.")
    
    masstrace_centroid_mz = np.array(masstrace_centroid_mz) * charge
    # Calculate mass differences (up to the first 5 differences)
    md = np.diff(masstrace_centroid_mz[:6])  # Limits to first 6 m/z values

    # Define acceptable ranges
    range1 = (0.9964, 1.0096)
    range2 = (0.9888, 1.0081)

    # Class 0
    if range1[0] < md[0] <= range1[1] and range2[0] < md[1] <= range2[1]:
        return 0

    # Class 1
    if len(md) >= 3:
        if range1[0] <= md[1] <= range1[1] and range2[0] < md[2] <= range2[1]:
            return 1

    # Class 2
    if len(md) >= 4:
        if range1[0] <= md[2] <= range1[1] and range2[0] < md[3] <= range2[1]:
            return 2

    # Class 3
    if len(md) >= 5:
        if range1[0] <= md[3] <= range1[1] and range2[0] < md[4] <= range2[1]:
            return 3

    # If none of the conditions are met
    return None

# Example Usage
if __name__ == "__main__":
    masstrace_centroid_mz = [100.0, 100.99, 101.98, 102.97, 103.96, 104.95]
    charge = 1
    label = feature_classifier(masstrace_centroid_mz, charge)
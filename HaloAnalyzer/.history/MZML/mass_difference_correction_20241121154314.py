import numpy as np
import pandas as pd

def feature_classifier_1(masstrace_centroid_mz:list):
    """
    class feature based on the mass difference between the centroid m/z values of the masstrace.
    """
    md = np.diff(masstrace_centroid_mz[:6])
    print(md)
    print(md.shape[0])
    if (0.9964<md[0]<=1.0096) and (0.9888<md[1]<=1.0081):
        return 0
    else:
        if  0.9964 <= md[1] <=1.0096:
            if len(masstrace_centroid_mz)<=3:
                pass
            else:
                if (0.9888<md[2]<=1.0081):
                    return 1
                else:
                    if 0.9964 <= md[2] <=1.0096:
                        if len(masstrace_centroid_mz) <=4:
                            pass
                        else:
                            if (0.9888<md[3]<=1.0081):
                                return 2
                            else:
                                if 0.9964 <= md[3] <=1.0096:
                                    if len(masstrace_centroid_mz) <=5:
                                        pass
                                    else:
                                        if (0.9888<md[4]<=1.0081):
                                            return 3
                                        else:
                                            pass
                                else:
                                    pass
            
        else:
            if md.shape>2 and 0.9964 <= md[2] <=1.0096:
                if len(masstrace_centroid_mz) <=4:
                    pass
                else:
                    if (0.9888<md[3]<=1.0081):
                        return 2
                    else:
                        if 0.9964 <= md[3] <=1.0096:
                            if len(masstrace_centroid_mz) <=5:
                                pass
                            else:
                                if (0.9888<md[4]<=1.0081):
                                    return 3
                                else:
                                    pass
                        else:
                            pass
        
                
import numpy as np
import pandas as pd

def feature_classifier(masstrace_centroid_mz: list) -> int:
    """
    Classify features based on the mass difference between the centroid m/z values of the masstrace.

    Args:
        masstrace_centroid_mz (list): List of centroid m/z values.

    Returns:
        int: Classification label (0, 1, 2, 3) or None if no classification criteria are met.
    """
    if not isinstance(masstrace_centroid_mz, list):
        raise TypeError("masstrace_centroid_mz must be a list of m/z values.")
    

    # Calculate mass differences (up to the first 5 differences)
    md = np.diff(masstrace_centroid_mz[:6])  # Limits to first 6 m/z values

    # Define acceptable ranges
    range1 = (0.9964, 1.0096)
    range2 = (0.9888, 1.0081)

    # Class 0
    if range1[0] < md[0] <= range1[1] and range2[0] < md[1] <= range2[1]:
        return 0

    # Class 1
    if len(md) >= 4:
        if range1[0] <= md[1] <= range1[1] and range2[0] < md[2] <= range2[1]:
            return 1

    # Class 2
    if len(md) >= 5:
        if range1[0] <= md[2] <= range1[1] and range2[0] < md[3] <= range2[1]:
            return 2

    # Class 3
    if len(md) >= 6:
        if range1[0] <= md[3] <= range1[1] and range2[0] < md[4] <= range2[1]:
            return 3

    # If none of the conditions are met
    return None

# Example Usage
if __name__ == "__main__":
    # Example centroid m/z values
    masstrace_centroid_mz_examples = [
[383.170343078632, 383.6726734197791, 384.1714005245743, 384.6721496582031, 385.1698913574219],
[431.06813717406044, 432.06777550250644, 433.05527776231077],
[667.2826488207435, 668.286428280542, 669.2817951651463, 670.2834340083849],
[689.2671564231646, 690.2679645497504, 691.2651262591908, 692.2659286198574],
[709.4224106880561, 710.4360275412258, 711.4171142578125, 712.4203778815925, 713.4252258890897]

    ]

    for mz in masstrace_centroid_mz_examples:
        classification = feature_classifier_1(mz)
        print(f"m/z values: {mz} => Classification: {classification}")
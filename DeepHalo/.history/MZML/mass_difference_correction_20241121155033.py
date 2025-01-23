import numpy as np
import pandas as pd

def feature_classifier_1(masstrace_centroid_mz:list,charge):
    """
    class feature based on the mass difference between the centroid m/z values of the masstrace.
    """
    masstrace_centroid_mz = np.array(masstrace_centroid_mz)*charge
    md = np.diff(masstrace_centroid_mz[:6])
    print(md)
    # print(md.shape[0])
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
            if md.shape[0] >=3 and 0.9964 <= md[2] <=1.0096:
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
    # Example centroid m/z values
    masstrace_centroid_mz_examples = [
[338.57201850954766, 339.5787407344456, 340.5770263671875],
[355.1842666456225, 355.68643400581266, 356.1856553826164],
[369.0517980223988, 370.05539491199454, 371.05085758334417],
[899.593704374954, 900.6006469726562, 901.5779800457527, 902.6033935546875, 903.5866676356123, 904.5945643406305],
[1128.3133318688774, 1129.3129040199099, 1130.3131500272455, 1131.312233297738, 1132.3097602695548],

    ]
    charge = [
        1,
2,
1,
1,
1,

    ]

    for mz in masstrace_centroid_mz_examples:
        classification = feature_classifier_1(mz,charge)
        print(f"m/z values: {mz} => Classification: {classification}")
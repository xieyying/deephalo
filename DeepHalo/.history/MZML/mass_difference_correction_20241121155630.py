import numpy as np
import pandas as pd

def feature_classifier_1(masstrace_centroid_mz:list,charge:int) -> int:
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
[466.2101596701229, 467.21170142791124, 468.20599365234375, 469.2082242339077],
[484.22064208984375, 485.22238209025085, 486.21628909330616],
[507.23736572265625, 508.23779296875, 509.2320222134088, 510.23401342320057, 511.23135541793783],
[512.2494311304231, 513.25, 514.2461685278662],
[521.2506397043337, 522.2530809686939, 523.2467798051982],
[535.2664475853995, 536.2668692248557, 537.262072958605, 538.2644942283358, 539.2614400467497],
[758.2201727110544, 759.2181396484375, 760.2179495029676, 761.2186313302897],
[815.4202977692504, 816.4241758415955, 817.4169319552736],


    ]
    charge = [
 1,
1,
1,
1,
1,
1,
1,
1,

    ]
    for i in range(len(masstrace_centroid_mz_examples)):
        group = feature_classifier_1(masstrace_centroid_mz_examples[i],charge[i])
        print(f"{masstrace_centroid_mz_examples[i]}, Group: {group}")
        


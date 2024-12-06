import numpy as np
import pandas as pd

def feature_classifier_1(masstrace_centroid_mz:list):
    """
    class feature based on the mass difference between the centroid m/z values of the masstrace.
    """
    md = [
        masstrace_centroid_mz[1]-masstrace_centroid_mz[0],
        masstrace_centroid_mz[2]-masstrace_centroid_mz[1],
        masstrace_centroid_mz[3]-masstrace_centroid_mz[2],
        masstrace_centroid_mz[4]-masstrace_centroid_mz[3],
        masstrace_centroid_mz[5]-masstrace_centroid_mz[4],
    ]
    md = np.array(md)
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
[449.05677896696113, 450.05954491900695, 451.05313235185235]
[537.3450427017847, 538.3507406626351, 539.334941715679, 540.3359233388841, 541.3511779702831, 542.3508505013798]
[598.1200769987119, 599.120433904179, 600.1149120471778, 601.1172538017655, 602.1207222142585]
[620.0980683805549, 621.1019796985571, 622.0963493938318, 623.1011316032628]
[621.3347293067145, 622.3382940381881, 623.3344987472344, 624.3446757053732]
[724.1113039757955, 725.1160846805748, 726.1095407763491, 727.1130145439575]
[1133.26517576777, 1134.2764190241373, 1135.275146680196, 1136.2764156445148, 1137.2659569974423]
[1217.1991594749431, 1218.2057340267452, 1219.1989279780878, 1220.201843124489, 1221.2001113889978, 1222.200226820858]
[1814.295332914748, 1815.2989389962097, 1816.2970418510586, 1817.300304804402, 1818.2949798112209, 1819.2969240524053]

    ]

    for mz in masstrace_centroid_mz_examples:
        classification = feature_classifier(mz)
        print(f"m/z values: {mz} => Classification: {classification}")
import numpy as np
import pandas as pd

def mass_difference_correction(masstrace_centroid_mz:list):
    """
    correct isotopes according to mass differences
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
        
                

    
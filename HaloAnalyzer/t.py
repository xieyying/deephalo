def calculate_zig_zag(I):
    """
    根据一个ROI中所有scan的分类结果，计算ZigZag score
    I:list，为一个ROI中所有scan的分类结果
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
    score = (4-8/N-zigzag)/(4-8/N)*100
    # score = (4-8/N-zigzag)/(4-8/N)*100
    return score
from collections import Counter
def roi_halo_evaluation(I):
    """
    根据一个ROI中所有scan的分类结果，判断该ROI为halo的概率
    I:list，为一个ROI中所有scan的分类结果
    """
    # Get the common classes in the ROI
    com_class = list(Counter(I).keys())
    print('com_class',com_class)
    # Determine the halo classification for the ROI
    if any(i in com_class for i in [0, 1, 2]):
        if len(com_class) == 1:
            halo_score = 100
            halo_sub_score = 100
            halo_class = 'halo'
            halo_sub_class = com_class[0]
        else:
            if {0, 1, 2}.issuperset(set(com_class)):
                halo_class = 'halo'
                halo_score = 100
                halo_sub_class =max(Counter(I).items(), key=lambda x: x[1])[0]
                halo_sub_score = calculate_zig_zag(I)
            else:
                I_new = [1 if i in [0,1,2] else 0 for i in I]
                print(I_new)
                if I_new.count(1) == I_new.count(0):
                    max_class = 1
                else:
                    max_class = max(Counter(I_new).items(), key=lambda x: x[1])[0]

                halo_class = ['halo' if max_class == 1 else 'non-halo'][0]
                if halo_class == 'halo':
                    halo_score = calculate_zig_zag(I_new)
                else:
                    halo_class = 'halo'
                    halo_score = 100-calculate_zig_zag(I_new)

                halo_sub_class = "None"
                halo_sub_score = "None"
    else:
        halo_class = 'non-halo'
        halo_score = 0
        halo_sub_class = 'None'
        halo_sub_score = 'None'
    return halo_class,halo_score,halo_sub_class,halo_sub_score


I = [2,3,3]
halo_class,halo_score,halo_sub_class,halo_sub_score =roi_halo_evaluation(I)
print(halo_class,halo_score,halo_sub_class,halo_sub_score)

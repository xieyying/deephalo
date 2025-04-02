from collections import Counter

def calculate_zig_zag(I):
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


def score_scans(I):
    """
    Determine the probability of an ROI being a halo based on the classification results of all scans in the ROI
    """
    # Get the common classes in the ROI
    com_class = list(Counter(I).keys())
    counter = Counter(I)
    # Calculate the ratio of 0,1,2 in I
    scan_based_halo_ratio = sum(1 for i in I if i in {0, 1, 2}) / len(I)

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
                    scan_based_halo_sub_score = calculate_zig_zag(I) * scan_based_halo_sub_class_ratio
                else:
                    scan_based_halo_sub_score = scan_based_halo_ratio
            else:
                I_new = [1 if i in [0,1,2] else 0 for i in I]
                if len(I) > 2:
                    scan_based_halo_score = calculate_zig_zag(I_new) * scan_based_halo_ratio
                else:
                    scan_based_halo_score = scan_based_halo_ratio
                scan_based_halo_sub_class = "None"
                scan_based_halo_sub_score = "None"
    else:
        scan_based_halo_class = 'non-halo'
        scan_based_halo_score = 0
        scan_based_halo_sub_class = 'None'
        scan_based_halo_sub_score = 'None'

    return scan_based_halo_class,scan_based_halo_score,scan_based_halo_sub_class,scan_based_halo_sub_score,scan_based_halo_ratio

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

def overall_score(scan_I, feature_g):
    """
    Calculate the overall score based on scan and feature scores.
    """
    # Calculate the overall score
    overall_score = 0.3 * (0) + 0.3 * (feature_g / 100) + 0.4 * (scan_I * feature_g / 10000)
    return overall_score


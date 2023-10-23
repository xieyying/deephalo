import pandas as pd
from collections import Counter


def halo_evaluation(file_name):
    # Read the input CSV file into a Pandas DataFrame
    df = pd.read_csv(file_name)
    # Get the unique ROIs in the DataFrame
    rois = df['roi'].unique()
    # Create an empty DataFrame to store the evaluation results
    roi_halo_eval = pd.DataFrame()

    # Loop over each ROI
    for roi in rois:

        # Get the data for the current ROI
        df_roi = df[df['roi'] == roi]
        mz = df_roi['mzmean'].tolist()[0]
        rt_sencond_start = df_roi['rt'].tolist()[0]
        rt_sencond_end = df_roi['rt'].tolist()[-1]
        rt_minute_start = rt_sencond_start/60
        rt_minute_end = rt_sencond_end/60
        roi_a0_int = df_roi['precursor_a0_ints'].sum()
        roi_a1_int = df_roi['precursor_a1_ints'].sum()
        roi_a2_int = df_roi['precursor_a2_ints'].sum()
        roi_a3_int = df_roi['precursor_a3_ints'].sum()
        index = df_roi['index'].tolist()

        
        # Get and Convert the hydro_classes to integers
        I= df_roi['hydro_class'].tolist()
        I = [int(i.split("'")[1]) for i in I]

        # Get the common hydrophobicity classes in the ROI
        com_class = list(Counter(I).keys())

        # Determine the halo classification for the ROI
        if any(i in com_class for i in [0, 1, 2]):
            if len(com_class) == 1:
                halo_score = 100
                halo_sub_score = 100
                halo_class = 'halo'
                halo_sub_class = com_class[0]
            else:
                if {0, 1, 2}.issuperset(set(com_class)):
                # if ((0 in com_class) and (1 in com_class)) or ((0 in com_class) and (2 in com_class)) or ((1 in com_class) and (2 in com_class)) or ((0 in com_class) and (1 in com_class) and (2 in com_class)):

                    halo_class = 'halo'
                    halo_score = 100
                    halo_sub_class =max(Counter(I).items(), key=lambda x: x[1])[0]
                    halo_sub_score = calculate_zig_zag(I)
                else:
                    I_new = [1 if i in [0,1,2] else 0 for i in I]
                    
                    # if the number of 1 in I is equal to the number of 0 in I, then the max_class is 1
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
            # Add the evaluation results for the ROI to the output DataFrame
            roi_halo_eval = pd.concat([roi_halo_eval, pd.DataFrame({'roi': [roi], 'mz': [mz], 'rt_second_start': [rt_sencond_start], 
                                                                    'rt_second_end': [rt_sencond_end], 'rt_minute_start': [rt_minute_start], 
                                                                    'rt_minute_end': [rt_minute_end], 'roi_a0_intensity': [roi_a0_int],
                                                                    'roi_a1_intesity': [roi_a1_int], 'roi_a2_intensity': [roi_a2_int], 
                                                                    'roi_a3_intensity': [roi_a3_int], 'roi_index': [index], 
                                                                    'halo_class': [halo_class], 'halo_score  %': [halo_score], 
                                                                    'halo_sub_class': [halo_sub_class], 'halo_sub_score': [halo_sub_score]})],
                                                                      ignore_index=True)
    
    # Save the evaluation results to a CSV file
    roi_halo_eval.to_csv(file_name.split('.csv')[0] + '_halo_eval.csv', index=False)
    
    # Return the DataFrame containing the evaluation results for all ROIs
    return roi_halo_eval

def calculate_zig_zag(I):
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
    return score

if __name__ == '__main__':
    # Specify the input CSV file and call the halo_evaluation function
    file = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\test1\test_mzml_prediction\target.csv'
    a = halo_evaluation(file)
    # Print the resulting DataFrame
    print(a)

    # I = [1, 1, 1, 1, 0, 1]
    # # I = [2, 2, 2, 2, 4, 2]
    # # I = [0, 1, 1,1, 0]
    # # I = [0, 1, 0,1, 0]
    # # I = [1, 0]
    # print(calculate_zig_zag(I))
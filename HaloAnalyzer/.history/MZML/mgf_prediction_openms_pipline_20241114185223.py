
import sys
sys.path.append(r'\Users\xyy\Desktop\python\HaloAnalyzer')

from pyteomics import mzml ,mgf
import numpy as np
import pandas as pd
from molmass import Formula
import tensorflow as tf
import os
from HaloAnalyzer.Model import model_build
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from keras.models import load_model

# your code here
def formula_clf(formula_dict,type=None) :
    """
    Returns a classifier based on the formula given.
    """

    #根据分子式，判断是否可训练
    if formula_dict.get('H') == None or formula_dict.get('C') == None:
        trainable = 'no'
    elif formula_dict.get('H') < 1 or formula_dict.get('C') < 1:
        trainable = 'no'
    elif formula_dict.get('S') != None and formula_dict.get('S') > 4:
        trainable = 'no'
    elif formula_dict.get('Se') != None and formula_dict.get('Se') > 1:
        trainable = 'no'
    else:
        trainable = 'yes'

    #根据分子式，判断类别
    if type == 'hydro':
        group = 7
    elif ('Br' in formula_dict.keys()) and ('Cl' in formula_dict.keys()):
        if 'B' in formula_dict.keys() or 'Se' in formula_dict.keys() or 'Fe' in formula_dict.keys():
            group = 10
        else:
            group = 0
    elif ('Br' in formula_dict.keys()) or ('Cl' in formula_dict.keys()):
        if 'B' in formula_dict.keys() or 'Se' in formula_dict.keys() or 'Fe' in formula_dict.keys():
            group = 10
        elif ('Br' in formula_dict.keys()) and formula_dict['Br']>1:
            group = 0
        elif ('Cl' in formula_dict.keys()) and formula_dict['Cl']>3:
            group = 0        
        elif ('Br' in formula_dict.keys()) and formula_dict['Br']==1 :
            group = 1
        elif ('Cl' in formula_dict.keys()) and formula_dict['Cl']==3:
            group = 1
        elif ('Cl' in formula_dict.keys()) and formula_dict['Cl']>=1:
            group = 2
    elif ('Se' in formula_dict.keys() ):
        group = 3

    elif ('B' in formula_dict.keys()):
        group = 4

    elif ('Fe' in formula_dict.keys()):
        group = 5
    elif ('S' in formula_dict.keys()):
        if formula_dict['S'] >=5:
            group = 8
        else:
            group = 6
    else:
        group = 6
    return trainable,group


#暂未修改
def get_charge(mz_list):
    mz_different = mz_list[1]-mz_list[0]
    charge_1 = abs(mz_different-1)
    charge_2 = abs(mz_different-0.5)
    charge_3 = abs(mz_different-0.33)
    if charge_1 < 0.02:
        charge = 1
    elif charge_2 < 0.02:
        charge = 2
    elif charge_3 < 0.02:
        charge = 3
    else:
        charge = 0
    return charge


def get_cal_mz(formula,mz,ints):
    # get mz_max of formula isotopic pattern
    a = Formula(formula+'H').spectrum().asdict()
    mz_monoisotope = a[0]            

    mz_list1 = mz[np.abs(mz-mz_monoisotope)<0.3]
    ints_list1 = ints[np.abs(mz-mz_monoisotope)<0.3]
    if len(mz_list1) != 0:
        precursor_mz = mz_list1[0]
        charge_t =1
        # print(precursor_mz)
    else:
        a = Formula(formula+"H"+'H').spectrum().asdict()
        mz_monoisotope = a[0]
        mz_list1 = mz[np.abs(mz-mz_monoisotope/2)<0.3]
        ints_list1 = ints[np.abs(mz-mz_monoisotope/2)<0.3]
        if len(mz_list1) != 0:
            precursor_mz = mz_list1[0]
            # print(precursor_mz)
            charge_t =2
        else:
            a = Formula(formula+"H"+'H'+'H').spectrum().asdict()
            mz_monoisotope= a[0]
            mz_list1 = mz[np.abs(mz-mz_monoisotope/3)<0.3]
            ints_list1 = ints[np.abs(mz-mz_monoisotope/3)<0.3]
            if len(mz_list1) != 0:
                precursor_mz = mz_list1[0]
                # print(precursor_mz)
                charge_t =3
            else:
                a = (Formula(formula)-Formula('H')).spectrum().asdict()
                mz_monoisotope = a[0]
                mz_list1 = mz[np.abs(mz-mz_monoisotope)<0.3]
                ints_list1 = ints[np.abs(mz-mz_monoisotope)<0.3]
                if len(mz_list1) != 0:
                    precursor_mz = mz_list1[0]
                    # print(precursor_mz)
                    charge_t =1
                else:
                    # print('more than 3 charges')
                    precursor_mz = 0
                    charge_t =0
                    
    return precursor_mz, mz_monoisotope,charge_t


def isotope_processing(df, mz_list_name = 'mz_list', inty_list_name = "inty_list"):
    """
    Process DataFrame and make it ready for halo model inputs
    """
    # get the mz_list and inty_list
    mz_list = df[mz_list_name].values
    print(type(mz_list[0]),mz_list[0])
    m2_m1 = [i[2] - i[1] for i in mz_list]
    m1_m0 = [i[1] - i[0] for i in mz_list]
    
    # Ensure all lists in inty_list have 7 elements
    inty_list = [i + [0]*(7-len(i)) for i in df[inty_list_name].tolist()]
    
    # Convert inty_list to a DataFrame
    inty_df = pd.DataFrame(inty_list)
    
    # Normalize each row by its max value
    inty_df = inty_df.div(inty_df.max(axis=1), axis=0)
    
    # Assign new columns to df
    df['m2_m1'] = m2_m1*df['charge']
    df['m1_m0'] = m1_m0*df['charge']
    for i in range(7):
        df[f'p{i}_int'] = inty_df[i].values
    return df

class mgf_pred():
    def __init__(self,file) -> None:
        self.path = file
        self.spectra = mgf.MGF(file)


    def get_df(self):
        halo_formula = []
        
        mgf_halo = []
        
        df = pd.DataFrame()
        m0 = []
        mz_list = []
        inty_list = []
        charge = []
        true_class = []
        cal_m0 = []
        formula_ = []
        scan = []
        trainable = []

        for spectrum in self.spectra:
            # formula = spectrum['params']['ch$formula']
            try:
                formula = spectrum['params']['ch$formula']
                name = spectrum['params']['ch$name']

            except:
                # for myxo
                formula = spectrum['params']['formula']
                name = spectrum['params']['compound_name']
      
            true_class_ = formula_clf(Formula(formula).composition().dataframe().to_dict()['Count'])[1]
            trainable_ = formula_clf(Formula(formula).composition().dataframe().to_dict()['Count'])[0]

            a = Formula(formula).spectrum().dataframe()
            cal_m0_ = a['m/z'].tolist()[0]          
          
            mz = spectrum['m/z array'].tolist()
            ints = spectrum['intensity array'].tolist()
            
            try:
                scan_ = spectrum['params']['scan']
            except:
                scan_ = 0
            
            if len(mz) <3:
                continue
            try:
                charge_ = spectrum['params']['charge'][0]

            except:
                charge_ = get_charge(mz)
            formula_.append(formula)
            m0.append(mz[0])
            mz_list.append(mz)
            inty_list.append(ints)
            charge.append(charge_)
            true_class.append(true_class_)
            cal_m0.append(cal_m0_)
            scan.append(scan_)
            trainable.append(trainable_)
            
            
                
        df['scan'] = scan
        df['mz_list'] = mz_list
        df['inty_list'] = inty_list
        df['charge'] = charge
        df['m0'] = m0
        df['cal_m0'] = cal_m0
        df['formula'] = formula_
        df['true_class'] = true_class
        df['trainable'] = trainable
        
            
        return df


def add_predict(df,model_path,features_list):
    
    clf = tf.keras.models.load_model(model_path)

    #加载特征
    querys = df[features_list].values
    querys = querys.astype('float32')
    #对特征进行预测
    res = clf.predict(querys)
    classes_pred = np.argmax(res, axis=1)
    #将预测结果添加到df_features中
    df['class_pred'] = classes_pred
    # 求 res中第0，1，2和元素的和
    res_sum = np.sum(res[:,0:3],axis=1)
    #如果其大于第3，4，5，6，7个元素中最大的，则halo为1，否则为0

    halo = [1 if res_sum[i] > np.max(res[i, 3:]) else 0 for i in range(len(res_sum))]
    df['halo'] = halo
    df['halo_pred'] = res.tolist()
    return df

def confusion_show(df):
        # # Compute confusion matrix
        Y_val = df['true_class'].astype('int').values
        y_pred = df['class_pred'].astype('int').values
        #y_pred中的7换为6
        y_pred[y_pred==7] = 6
        print(type(Y_val),type(y_pred))
        # Merge classes 0, 1, 2 into a new class
        # Y_val_merged = Y_val.copy()
        # Y_val_merged[(Y_val_merged == 0) | (Y_val_merged == 1) | (Y_val_merged == 2)] = 7

        # y_pred_merged = y_pred.copy()
        # y_pred_merged[(y_pred_merged == 0) | (y_pred_merged == 1) | (y_pred_merged == 2)] = 7
         
        # Y_val = Y_val_merged
        # y_pred = y_pred_merged

        cm = confusion_matrix(Y_val, y_pred)


        # Compute recall and precision for each class
        report = classification_report(Y_val, y_pred, output_dict=True,zero_division=1)
        recalls = [report[str(i)]['recall'] if str(i) in report else 0  for i in range(8)]
        precisions = [report[str(i)]['precision'] if str(i) in report else 0 for i in range(8)]
        F1_sore = [report[str(i)]['f1-score'] if str(i) in report else 0 for i in range(8)]

        # Plot the confusion matrix
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # Plot the confusion matrix
        ConfusionMatrixDisplay.from_predictions(Y_val, y_pred, ax=axs[0, 0], cmap=plt.cm.terrain)
        axs[0, 0].set_title('Classifier')
        # Plot the precision bar chart below the confusion matrix
        
        colors = np.random.rand(len(precisions), 3)
        axs[1, 0].bar(np.arange(len(precisions)), precisions, color=colors)
        axs[1, 0].set_title('Precision')
        axs[1, 0].set_xlabel('Class')
        axs[1, 0].set_ylabel('Precision')
        #将数值标注在图上
        for i, v in enumerate(precisions):
            axs[1, 0].text(i - 0.25, v + 0.01, str(round(v, 3)), color='black', fontweight='bold')

        # Plot the recall bar chart to the right of the confusion matrix
        
        axs[0, 1].barh(np.arange(len(recalls)), recalls, color=colors)
        axs[0, 1].set_title('Recall')
        axs[0, 1].set_xlabel('Recall')
        axs[0, 1].set_ylabel('Class')
        #将数值标注在图上
        for i, v in enumerate(recalls):
            axs[0, 1].text(v + 0.01, i + .25, str(round(v, 3)), color='black', fontweight='bold')
        axs[0, 1].invert_yaxis()  # Reverse the y-axis so class 0 is on top
        axs[0, 1].set_xlim(0, 1)  # Set the x-axis range to 0-1

        # Plot the f1-score bar chart below the recall bar chart
        axs[1, 1].bar(np.arange(len(F1_sore)), F1_sore, color=colors)
        axs[1, 1].set_title('F1-score')
        axs[1, 1].set_xlabel('Class')
        axs[1, 1].set_ylabel('F1-score')
        #将数值标注在图上
        for i, v in enumerate(F1_sore):
            axs[1, 1].text(i - 0.25, v + 0.01, str(round(v, 3)), color='black', fontweight='bold')
        axs[1, 1].set_ylim(0, 1)
        # axs[1, 1].set_xlim(0, 1)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    
    feature_list = [
        "p0_int",
        "p1_int",
        "p2_int",
        "p3_int",
        "p4_int",
        # "p5_int",
        "m2_m1",
        "m1_m0",        
    ]

    model = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms_noClFe\2M_fake_molecules\noisy_mz_001_inty_03_new_data\trained_models\pick_halo_ann.h5'
    model = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms_noClFe\2M_fake_molecules\200_trails_mz_noise_0.001_inty_0.04_5_peaks\trained_models\pick_halo_ann.h5'

    path = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\test_dataset\open_dataset\myxo'

    path = r'C:\Users\xyy\Desktop\test\test'  
    # path = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\training_validation_dataset\train_and_val\validation_data_noise_added\inty_0.07_0.006_mz_0.0018'
    # path = r'D:\workissues\manuscript\halo_mining\HaloAnalyzer\datasets\training_validation_dataset\train_and_val\validation_data_noise_added\inty_0.07_0.006_mz_0.0018\test'
    
    files = os.listdir(path)
    for file in files:
        f = os.path.join(path,file)
        # 以mgf结尾
        if not f.endswith('.mgf'):
            continue
        print(file)
        a= mgf_pred(f)
        df = a.get_df()
        # df.to_csv(rf'C:\Users\xyy\Desktop\test\tem\{file[:-4]}_features.csv',index=False)
        df = isotope_processing(df)
        df = add_predict(df,model,feature_list)
        df.to_csv(rf'C:\Users\xyy\Desktop\test\tem\{file[:-4]}_features.csv',index=False)
        n = 0
        tn=0
        wrong_formula = []
        mgf_patterns_right =[]
        mgf_extracted_wrong = []
        
        pred_wrong = pd.DataFrame()
        for i in range(len(df)):
            t = df['true_class'].tolist()[i]
            p = df['class_pred'].tolist()[i]
            # print(t,p)
            if t != p:
                wrong_formula.append(df['formula'].tolist()[i])
                #打印这一行
                # print(df.iloc[i, :] )
                # print(df.iloc[[i]])
                pred_wrong = pd.concat([pred_wrong,df.iloc[[i]]])
                
                print(t,p)
                n+=1
          
            if t ==0:
                tn+=1
        pred_wrong.to_csv(rf'C:\Users\xyy\Desktop\test\tem\{file[:-4]}_pred_wrong.csv',index=False)
        # for s in mgf_patterns:
        #     if s['params']['ch$formula'] not in wrong_formula:
        #         mgf_patterns_right.append(s)
        # mgf_extracted = mgf.read(f)
        # print(mgf_extracted )
        # print(wrong_formula)
        # for sp in mgf_extracted:
        #     print(sp['params']['formula'])
         
        #     if sp['params']['ch$formula'] in wrong_formula:
        #         mgf_extracted_wrong.append(sp)
        # mgf.write(mgf_extracted_wrong,rf'C:\Users\xyy\Desktop\test\tem\{file[:-4]}_extracted_wrong.mgf')
                
        # print('----------------')
        
        # mgf.write(mgf_patterns_right,f[:-4]+'_patterns_right.mgf')
        # mgf.write(mgf_extracted_wrong,f[:-4]+'_extracted_wrong.mgf')
        
        print('total: ', len(df), '; wrong', n,'multi_halo: ',tn)
        confusion_show(df)
        # print(f,'done')
        
    # #adding prediction to the files
    # for file in files:
    #     if file.endswith('.csv'):
    #         f = os.path.join(path,file)
    #         df = pd.read_csv(f)
    #         df_ = add_predict(df,model,feature_list)
    #         df_.to_csv(f,index=False)
        
            
            


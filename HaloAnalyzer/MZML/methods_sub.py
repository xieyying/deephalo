import pymzml
import pandas as pd
from asari.peaks import *
from asari.default_parameters import PARAMETERS
from collections import Counter

class ROIs:
    def __init__(self) -> None:
        self.rois = []
        self.max_roi = 0
    
    def update(self,update_dict,roi_precusor_error = 20, gap_scans = 3):

        #如果self.rois为空，则将update_dict['precursor']作为一个新的roi
        if self.rois == []:
            self.rois.append({
                'id_roi':self.max_roi,
                'mz_mean':update_dict['precursor'],
                'MS1_index':[update_dict['MS1']],
                'counter_list':[update_dict['MS1_counter']],
                'roi_ms2_index':[update_dict['MS2']],
                'left_base':update_dict['MS1_counter'],
                'right_base':update_dict['MS1_counter'],
            })
            self.max_roi += 1
        
        else:
            add_new = False
            for roi in self.rois:
                delt_mz = abs(roi['mz_mean'] - update_dict['precursor'])/update_dict['precursor']*1e6
                delt_scan = abs(roi['counter_list'][-1] - update_dict['MS1_counter'])
                if delt_mz <= roi_precusor_error and delt_scan <= gap_scans:

                    roi['mz_mean'] = (roi['mz_mean'] * len(roi['MS1_index']) + update_dict['precursor'])/(len(roi['MS1_index']) + 1)
                    if update_dict['MS1_counter']  not in roi['counter_list']:
                        roi['MS1_index'].append(update_dict['MS1'])
                        roi['counter_list'].append(update_dict['MS1_counter'])
                    roi['roi_ms2_index'].append(update_dict['MS2'])
                    #更新roi的左右边界
                    roi['left_base'] = min(roi['left_base'],update_dict['MS1_counter'])
                    roi['right_base'] = max(roi['right_base'],update_dict['MS1_counter'])
                    add_new = True
                    # break
            if not add_new:
                # print('new')
                self.rois.append({
                    'id_roi':self.max_roi,
                    'mz_mean':update_dict['precursor'],
                    'MS1_index':[update_dict['MS1']],
                    'counter_list':[update_dict['MS1_counter']],
                    'roi_ms2_index':[update_dict['MS2']],
                    'left_base':update_dict['MS1_counter'],
                    'right_base':update_dict['MS1_counter'],
                })
                self.max_roi += 1

    #过滤掉MS1_index长度小于min_points的roi
    def filter(self,min_points):
        self.rois = [roi for roi in self.rois if len(roi['MS1_index']) > min_points]
        df = pd.DataFrame(self.rois)
        return df

    def merge(self,merge_precursor_error = 20, merge_gap_scans = 3):
        #将self.rois按照mz_mean从小到大排序
        self.rois = sorted(self.rois,key=lambda x:x['mz_mean'])

        for i in range(len(self.rois)-1):
            delt_mz = abs(self.rois[i]['mz_mean'] - self.rois[i+1]['mz_mean'])/self.rois[i]['mz_mean']*1e6
            delt_scan = abs(self.rois[i]['counter_list'][-1] - self.rois[i+1]['counter_list'][0])
            if delt_mz <= merge_precursor_error and delt_scan <= merge_gap_scans:
                self.rois[i]['mz_mean'] = (self.rois[i]['mz_mean'] * len(self.rois[i]['MS1_index']) + self.rois[i+1]['mz_mean'] * len(self.rois[i+1]['MS1_index']))/(len(self.rois[i]['MS1_index']) + len(self.rois[i+1]['MS1_index']))
                self.rois[i]['MS1_index'] = self.rois[i]['MS1_index'] + self.rois[i+1]['MS1_index']
                self.rois[i]['counter_list'] = self.rois[i]['counter_list'] + self.rois[i+1]['counter_list']
                self.rois[i]['roi_ms2_index'] = self.rois[i]['roi_ms2_index'] + self.rois[i+1]['roi_ms2_index']
                self.rois[i]['left_base'] = min(self.rois[i]['left_base'],self.rois[i+1]['left_base'])
                self.rois[i]['right_base'] = max(self.rois[i]['right_base'],self.rois[i+1]['right_base'])
                self.rois.pop(i+1)
                self.merge()

    def get_roi_df(self):
        df = pd.DataFrame(self.rois)
        return df

def feature_extractor(file_name: str,para) -> pd.DataFrame:
    """
    Extract features from an mzML file and return a DataFrame containing the feature information.

    Args:
        file_name: The path to the mzML file.

    Returns:
        A DataFrame containing feature information, including the quality-time coordinates of the features, peak area, peak height, and other information.

    this function is modified from asari
    """
    ms_expt = pymzml.run.Reader(file_name)

    # set parameters
    parameters = PARAMETERS
    parameters['min_prominence_threshold'] = int(para['min_prominence_threshold'] * parameters['min_peak_height'])

    # extract mass tracks

    list_mass_tracks1 = extract_massTracks_(ms_expt, mz_tolerance_ppm=para['mz_tolerance_ppm'], min_intensity=para['min_intensity'], min_timepoints=para['min_timepoints'], min_peak_height=para['min_peak_height'])

    # reformat mass tracks to fit the input format of batch_deep_detect_elution_peaks
    list_mass_tracks = [{'id_number': i, 'mz': x[0], 'intensity': x[1]} for i, x in enumerate(list_mass_tracks1["tracks"])]

    # extract features
    features = batch_deep_detect_elution_peaks(list_mass_tracks, number_of_scans=len(list_mass_tracks1['rt_numbers']), parameters=parameters)

    df_features = pd.DataFrame(features)

    return df_features

def MS1_MS2_connected(spectra,mzml_dict):
    vendor = mzml_dict['vendor']
    precursor_error = mzml_dict['precursor_error']

    """
    将MS1和MS2连接起来，返回一个dataframe，包含MS2的index以及与之对应的MS1的index

    质谱DDA采集模式下，信号采集顺序为MS1,而后是与此MS1相对应的一个或多个MS2(根据采集设置而定)

    在waters采集中MS1的function=1

    """
 
    MS1_MS2_connected = {}
    MS1_MS2_connected['MS1'] = []
    MS1_MS2_connected['MS2'] = []
    MS1_MS2_connected['precursor'] = []
    MS1_MS2_connected['rt'] = []
    MS1_MS2_connected['MS1_counter'] = []
    MS1_MS1_index = []
    MS1_rt = []
    MS1_counter_list = []
    MS1_counter=-1
    
    for s in spectra:
        try:
            if (s['ms level'] == 1 and s['id'].split(' ')[0] == "function=1" and vendor == 'waters') or (s['ms level'] == 1 and vendor != 'waters'):
                mz_list = s['m/z array']
                ints_list = s['intensity array']
                MS1_counter += 1
                MS1_MS1_index.append(s['index'])
                MS1_rt.append(s['scanList']['scan'][0]['scan start time'])
                MS1_counter_list.append(MS1_counter)
                
            elif s['ms level'] == 2:
                precursor_mz_source = s['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z']
                #找到mz中与precursor_mz相差在0.3的所有mz_list中最高的峰
                mz_list1 = mz_list[np.abs(mz_list-precursor_mz_source)<precursor_error]
                ints_list1 = ints_list[np.abs(mz_list-precursor_mz_source)<precursor_error]
                precursor_mz = mz_list1[np.argmax(ints_list1)]

                MS1_MS2_connected['MS1_counter'].append(MS1_counter_list[-1])
                MS1_MS2_connected['rt'].append(MS1_rt[-1])
                MS1_MS2_connected['MS1'].append(MS1_MS1_index[-1])

                MS1_MS2_connected['MS2'].append(s['index'])
                MS1_MS2_connected['precursor'].append(precursor_mz)
            else:
                continue
        except:
            continue

    # transfer MS1_MS2_connected to a dataframe
    MS1_MS2_connected = pd.DataFrame(MS1_MS2_connected)

    return MS1_MS2_connected

#待更新减去质谱中的基线
def fliter_mzml_data(ms1_spectra,min_intensity):
    rt = ms1_spectra['scanList']['scan'][0]['scan start time']*60
    mz = ms1_spectra['m/z array']
    intensity = ms1_spectra['intensity array']
    #只保留intensity大于min_intensity的峰
    mz = mz[intensity>min_intensity]
    intensity = intensity[intensity>min_intensity]
    return rt,mz,intensity

#误差范围也需要同步传递
def get_mz_max(mz,intensity,target_mz):
    #获取mz中与target_mz相差在0.02的所有mz
    mz_list1 = mz[np.abs(mz-target_mz)<0.02]
    ints_list1 = intensity[np.abs(mz-target_mz)<0.02]
    #获取mz_list1中intensity最大的mz
    mz_max1 = mz_list1[np.argmax(ints_list1)]
    #获取mz_max1对应的intensity
    # intensity_max1 = intensity[np.argmax(intensity[np.abs(mz-target_mz)<0.02])]
    intensity_max1 = ints_list1.max()
    
    #获取mz中与target_mz相差在-3.1和+3.1的所有mz
    mz_list2_index = pd.Series(mz).between((target_mz-3.1),(target_mz+3.1))
    mz_list2 = mz[mz_list2_index]
    ints_list2 = intensity[mz_list2_index]
    #获取mz_list2中intensity最大的mz
    mz_max2 = mz_list2[np.argmax(ints_list2)]
    #获取mz_max2对应的intensity
    # intensity_max2 = intensity[np.argmax(intensity[mz_list2_index])]
    intensity_max2 = ints_list2.max()
    #以字典的形式返回
    return {'mz_list1':mz_list1,'ints_list1':ints_list1,'mz_max1':mz_max1,'intensity_max1':intensity_max1,'mz_list2':mz_list2,'ints_list2':ints_list2,'mz_max2':mz_max2,'intensity_max2':intensity_max2}

#暂未修改
def get_charge(mz_list,ints_list,intensity_max):
    mz_charge_list = mz_list
    ints_charge_list = ints_list/intensity_max
    #选取mz_charge_list中强度最大的前5个峰
    #若不足五个峰，则选取全部
    if len(mz_charge_list) >= 5:
        a = mz_charge_list[ints_charge_list.argsort()[-5:][::-1]]
    else:
        #按照强度顺序排列
        a = mz_charge_list[ints_charge_list.argsort()[::-1]]
    b = []
    for i in a:
        for j in a:
            b.append(i-j)
    b = np.array(b)
    
    b = b.reshape(len(a),len(a))
    # print(b)a
    for i in range(0,len(b)):
        for j in range(0,len(b)):
            if abs(b[i][j] - 1) < 0.02:
                b[i][j] = 1
            elif abs(b[i][j] - 0.5) < 0.02:
                b[i][j] = 0.5
            elif abs(b[i][j] - 0.33) < 0.02:
                b[i][j] = 0.33
            else:
                b[i][j] = 0
    # print(b)
    c = {}
    for i in range(0,len(b)):
        for j in range(0,len(b)):
            if b[i][j] in c:
                c[b[i][j]] += 1
            else:
                c[b[i][j]] = 1
    # print(c)
    d = 0
    for i in c:
        if i != 0:
            if c[i] > d:
                d = c[i]
                e = i
    if d == 0:
        return 0
    if e == 1:
        return 1
    elif e == 0.5:
        return 2
    elif e == 0.33:
        return 3
    else:
        return 0

def get_one_isotopologue(mz_list,ints_list,mz_max,charge,delta_mz,error=0.02):
    
    """
    以最高峰为基准，获取与其相差delta_mz的同位素峰的mz和intensity;
    考虑电荷
    允许误差范围为error
    """

    mz = [i for i, x in enumerate(mz_list) if (mz_max+(delta_mz-error)/charge) <= x <= (mz_max+(delta_mz+error)/charge)]
    if len(mz) != 0:
        ints = [ints_list[i] for i in mz]
        mz = mz_list[max(mz, key=lambda i: ints_list[i])]
        ints = max(ints)
    else:
        mz = 0
        ints = 0
    return mz,ints

def get_isotoplogues(mz_max,mz_list,ints_list,charge):
    """
    以最高峰为基准，获取同位素峰的mz和intensity;

    """
    mz_b3,ints_b3 = get_one_isotopologue(mz_list,ints_list,mz_max,charge,-3)
    mz_b2,ints_b2 = get_one_isotopologue(mz_list,ints_list,mz_max,charge,-2)
    mz_b1,ints_b1 = get_one_isotopologue(mz_list,ints_list,mz_max,charge,-1)
    mz_a0,ints_a0 = get_one_isotopologue(mz_list,ints_list,mz_max,charge,0)
    mz_a1,ints_a1 = get_one_isotopologue(mz_list,ints_list,mz_max,charge,1)
    mz_a2,ints_a2 = get_one_isotopologue(mz_list,ints_list,mz_max,charge,2)
    mz_a3,ints_a3 = get_one_isotopologue(mz_list,ints_list,mz_max,charge,3)

    ints_b3,ints_b2,ints_b1,ints_a0,ints_a1,ints_a2,ints_a3 = ints_b3/ints_a0,\
    ints_b2/ints_a0,ints_b1/ints_a0,ints_a0/ints_a0,ints_a1/ints_a0,ints_a2/ints_a0,ints_a3/ints_a0
    
    # 强度小于0.01的intensity和mz都设置为0，防止背景信号影响
    if ints_b3 < 0.01:
        mz_b3 = 0
        ints_b3 = 0
    if ints_b2 < 0.01:
        mz_b2 = 0
        ints_b2 = 0
    if ints_b1 < 0.01:
        mz_b1 = 0
        ints_b1 = 0
    if ints_a1 < 0.01:
        mz_a1 = 0
        ints_a1 = 0
    if ints_a2 < 0.01:
        mz_a2 = 0
        ints_a2 = 0
    if ints_a3 < 0.01:
        mz_a3 = 0
        ints_a3 = 0

    #以字典的形式返回
    return {'mz_b3':mz_b3,'ints_b3':ints_b3,'mz_b2':mz_b2,'ints_b2':ints_b2,'mz_b1':mz_b1,'ints_b1':ints_b1,'mz_a0':mz_a0,'ints_a0':ints_a0,'mz_a1':mz_a1,'ints_a1':ints_a1,'mz_a2':mz_a2,'ints_a2':ints_a2,'mz_a3':mz_a3,'ints_a3':ints_a3}

def is_halo_isotopes(b_3,b_2,b_1,a0,a1,a2,a3):
    """
    根据六个个同位素峰的强度,判断是否为同位素峰
    判断主要依据卤化物的同位素峰强度的统计结果
    """
    # if a1>0.02:
    #     if b_2 == 0:
    #         if b_1==0:
    #             is_isotope = 1
    #         else:
    #             if  b_1 > 0.5:
    #                 is_isotope = 1
    #             else:
    #                 is_isotope = 0
    #     else:
    #         if b_2>0.3:
    #             if b_1>0.02:
    #                is_isotope = 1
    #             else:
    #                 is_isotope = 0
    #         else:
    #             is_isotope = 0
    # else:
    #     is_isotope =0

    # return is_isotope
    if a1>0.02:
        if b_3 == 0:
            if b_2 == 0:
                is_isotope = 1
            elif b_2>0.04:
                is_isotope = 1
            else:
                is_isotope = 0
        else:
            if b_2 > 0.04:
                is_isotope = 1
            else:
                is_isotope = 0
    else:
        is_isotope =0
    return is_isotope
    
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

def roi_halo_evaluation(I):
    """
    根据一个ROI中所有scan的分类结果，判断该ROI为halo的概率
    I:list，为一个ROI中所有scan的分类结果
    """
    # Get the common classes in the ROI
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
                halo_class = 'halo'
                halo_score = 100
                halo_sub_class =max(Counter(I).items(), key=lambda x: x[1])[0]
                halo_sub_score = calculate_zig_zag(I)
            else:
                I_new = [1 if i in [0,1,2] else 0 for i in I]
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

if __name__ == "__main__":
    def ms2ms1_linked_ROI_identify(spectra,mzml_dict):
        mzml_dict['vendor'] = 'waters'
        mzml_dict['precursor_error'] = 0.3
        df = MS1_MS2_connected(spectra, mzml_dict)

        #将df中的每一行转为一个字典
        t = df.to_dict('records')

        rois = ROIs()
        for i in range(len(t)):
            rois.update(t[i])

        rois.merge()
        rois.filter(3)  # 这个参数应设置为可调参数

        df = rois.get_roi_df()
        return df
    from pyteomics import mzml

    spectra = mzml.read(r'J:\wangmengyuan\dataset\mzmls\Vancomycin.mzML')
    df = ms2ms1_linked_ROI_identify(spectra,{'vendor':'waters','precursor_error':0.3})
    print(df)
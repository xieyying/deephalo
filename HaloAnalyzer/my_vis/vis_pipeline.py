import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pyteomics import mzml
class PipelineVis:
    def __init__(self,mzml_path):
        self.mzml_path = mzml_path

    def load_base_data(self):
        #读取数据
        self.features = pd.read_csv(r'./test_mzml_prediction/features.csv')
        self.target = pd.read_csv(r'./test_mzml_prediction/target.csv')
        self.tic = pd.read_csv(r'./test_mzml_prediction/tic.csv')

        x_scale = alt.Scale(domain=(self.tic['scan'].min(),self.tic['scan'].max()))
        selection = alt.selection_point(fields=['class'], bind='legend')
        #将self.target中的intensity列转换为0-1之间的数
        self.target['intensity'] = self.target['intensity']/self.target['intensity'].max()

        #绘tic图
        bar_args = {'opacity': .8, 'binSpacing': 0}
        self.tic_chart = alt.Chart(self.tic).mark_bar(**bar_args).encode(
            alt.X('scan',scale=x_scale),
            y='tic',
            tooltip=['rt','tic'],
            
        ).interactive().properties(
            width=600,
            height=100,
        )
        
        #绘制点图
        base = alt.Chart(self.target)
        
        self.target_chart = base.mark_point().encode(
            alt.X('scan',scale=x_scale),
            y='mz',
            color='base_class',
            size = 'intensity',
            tooltip=['roi','scan','rt','mz','Charge','base_class','sub_class','hydro_class','intensity'],
            opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
        ).interactive().add_params(
            selection
        )
        
        self.target_chart_charge = base.mark_point().encode(
            alt.X('scan',scale=x_scale),
            y='mz',
            color='Charge',
            tooltip=['roi','scan','rt','mz','Charge','base_class','sub_class','hydro_class','intensity'],
            opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
        ).interactive().add_params(
            selection
        )
        

        self.target_combine = alt.vconcat(self.target_chart,self.tic_chart)
        
    def roi_vis(self,roi):


        #过滤target，找到roi对应的行
        target_roi = self.target[self.target['roi']==roi]

        #绘制点图
        self.target_chart_roi = (alt.Chart(target_roi).mark_point().encode(
            x=alt.X('scan',scale=alt.Scale(domain=(target_roi['scan'].min(),target_roi['scan'].max()))),
            y='intensity',
            color='base_class',
            tooltip=['scan','rt','mz','Charge','base_class','sub_class','hydro_class']
        ).interactive()).properties(
            width=600,
            height=250
        )


        scans = target_roi['scan']
        intensity = target_roi['intensity']
        #在scan的第一个值前加入比第一个值小1的值
        scans = scans.tolist()
        scans.insert(0,scans[0]-1)
        #在intensity的第一个值前加入0
        intensity = intensity.tolist()
        intensity.insert(0,0)
        #在scan的最后一个值后加入比最后一个值大1的值
        scans.append(scans[-1]+1)
        #在intensity的最后一个值后加入0
        intensity.append(0)
        #将新的scan和intensity转换为dataframe
        target_roi = pd.DataFrame({'scan':scans,'intensity':intensity})


        #绘制曲线图
        self.target_chart_roi_line = (alt.Chart(target_roi).mark_line().encode(
            #x轴为scan，y轴为intensity
            #x轴的范围为scan的最小值和最大值
            x=alt.X('scan',scale=alt.Scale(domain=(target_roi['scan'].min(),target_roi['scan'].max()))),
            y='intensity',
            # color='class',
            # tooltip=['scan','rt','mz','Charge','class']
        ).interactive())


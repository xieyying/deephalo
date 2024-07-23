import os
import altair as alt
import streamlit as st
from streamlit_elements import elements, mui
import pandas as pd


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

def show_pipeline_vis(pipeline_vis):
    if pipeline_vis != None:
        pipeline_vis.load_base_data()
        st.title('Pipeline Visualization')
        with elements("multiple_children"):
            mui.Button(
                mui.icon.DoubleArrow,
                pipeline_vis.mzml_path,
            )
        col1,col2 = st.columns(2)
        with col1:
            st.subheader('Pick_Halo_Target with TIC_Spectrum')
            st.altair_chart(pipeline_vis.target_chart ,use_container_width=True)
            st.subheader('Pick_Halo_Target with Charge')
            st.altair_chart(pipeline_vis.target_chart_charge ,use_container_width=True)
        with col2:
            st.subheader('ROI_Class_result')        
            roi_list = pipeline_vis.target['roi'].unique().tolist()
            roi_list.sort()
            roi = st.select_slider('Select ROI',roi_list)
            pipeline_vis.roi_vis(roi)
            st.altair_chart(pipeline_vis.target_chart_roi+pipeline_vis.target_chart_roi_line,use_container_width=True)


def main(pipeline_vis):
    #wide mode
    st.set_page_config(layout='wide')
    #网页标题
    st.title('Halo Mass Spectrometry Data Analysis')
    st.markdown('This application is a Streamlit dashboard for Halo Mass Spectrometry Data Analysis')
    st.markdown('---')
    st.sidebar.markdown('This application is a Streamlit dashboard for Halo Mass Spectrometry Data Analysis')
    st.sidebar.markdown('Please select the page you want to view')
    pages = st.sidebar.radio('Select page', ['Home', 'Pipeline visualization'])  
    if pages == 'Pipeline visualization':
        show_pipeline_vis(pipeline_vis)
        st.markdown('---')
    else:
        st.write('This is the home page of Halo Mass Spectrometry Data Analysis')
        st.write('Please select the page you want to view on the left')
        st.markdown('---')


if __name__ == '__main__':
    #从log文件中读取mzml_path

    project_path = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms\result'
    os.chdir(project_path)
    with open(r'test_mzml_prediction\log.txt','r') as f:
        mzml_path = f.read()
    pipeline_vis = PipelineVis(mzml_path)
    main(pipeline_vis)

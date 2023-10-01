import os
from my_vis.vis_pipeline import PipelineVis
from my_vis.vis_dataset import Dataset_vis
from my_vis.vis_model import Model_vis
from my_vis.intro import introduction
import tomli
import importlib_resources

import streamlit as st
from streamlit_elements import elements, mui

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


def show_model_vis(model_vis):
    if model_vis != None:
        st.title('Model Visualization')

        col1,col2,col3 = st.columns(3)
        with col1:
            st.subheader('Model Summary')
            st.image(model_vis.model_sum)

        with col2:    
            st.subheader('Model Training Parameters')
            st.write(model_vis.train_para)
        with col3:
            st.subheader('Confusion Matrix')
            st.pyplot(model_vis.cm_figure)

def show_dataset_vis(dataset_vis):
    if dataset_vis != None:
        st.title('Dataset Visualization')
        st.subheader('Dataset Key Composition')
        st.write(dataset_vis.Key_composition)

        col1, col2 ,col3= st.columns(3)

        with col1:
            st.subheader('trainable dataset composition')
            st.write(dataset_vis.Train_composition,use_container_width=True)
        with col2:
            st.subheader('mass distribution')
            st.write(dataset_vis.Mass_composition,use_container_width=True)

        with col3:
            st.subheader('group composition')
            st.write(dataset_vis.Class_composition,use_container_width=True)

def main(pipeline_vis,dataset_vis,model_vis):
    #wide mode
    st.set_page_config(layout='wide')
    #网页标题
    st.title('Halo Mass Spectrometry Data Analysis')
    # st.sidebar.title('Halo Mass Spectrometry Data Analysis')
    st.markdown('This application is a Streamlit dashboard for Halo Mass Spectrometry Data Analysis')
    st.markdown('---')
    st.sidebar.markdown('This application is a Streamlit dashboard for Halo Mass Spectrometry Data Analysis')
    st.sidebar.markdown('Please select the page you want to view')
    pages = st.sidebar.radio('Select page', ['Home', 'Dataset visualization', 'Model visualization','Pipeline visualization'])

    if pages == 'Dataset visualization':
        show_dataset_vis(dataset_vis)
        st.markdown('---')
    elif pages == 'Model visualization':
        show_model_vis(model_vis)
        st.markdown('---')    
    elif pages == 'Pipeline visualization':
        show_pipeline_vis(pipeline_vis)
        st.markdown('---')
    else:
        st.write('This is the home page of Halo Mass Spectrometry Data Analysis')
        st.write('Please select the page you want to view on the left')
        st.markdown('---')
        introduction()






if __name__ == '__main__':
    #从log文件中读取mzml_path
    file_path = importlib_resources.files('HaloAnalyzer') / 'config.toml'
    with open(file_path,'rb') as f:
        config = tomli.load(f)
    project_path = config['visualization']['path']
    os.chdir(project_path)
    with open(r'test_mzml_prediction\log.txt','r') as f:
        mzml_path = f.read()
    pipeline_vis = PipelineVis(mzml_path)
    dataset_vis = Dataset_vis()
    model_vis = Model_vis()
   
    main(pipeline_vis,dataset_vis,model_vis)

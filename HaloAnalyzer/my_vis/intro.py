import streamlit as st
import os
from PIL import Image
import importlib_resources
def introduction():
    #introduction
    st.subheader('Introduction')
    st.write('This is a web application for the visualization of the pipeline, dataset and model.')
    st.subheader('Implementation：')
    path = importlib_resources.files('HaloAnalyzer') / 'my_vis/intro.png'
    # path = r'C:\Users\Administrator\Desktop\HaloAnalyzer-0.1.0\HaloAnalyzer\my_vis\intro.png'
    image = Image.open(path)
    st.image(image, width=800)



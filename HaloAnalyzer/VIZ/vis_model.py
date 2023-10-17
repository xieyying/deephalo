import pickle
import tensorflow as tf
import numpy as np
import streamlit as st
class Model_vis:
    def __init__(self):
        #读取model的pickle文件
        self.cm_figure,self.train_para = pickle.load(open(r'./trained_models/pick_halo_ann_sum.pkl','rb'))
        
        self.model_sum = r'./trained_models/pick_halo_ann.png'




if __name__ == '__main__':
    model_vis = Model_vis()

    print(model_vis.train_para)
    print(model_vis.cm_figure)
    print('end')
import pickle
import streamlit as st
import altair as alt
import pandas as pd

class Dataset_vis:
    def __init__(self):
        #读取pickle文件
        msg1,msg2,msg3,msg4 = pickle.load(open(r'./train_dataset/dataset_statistics_customized.pkl','rb'))

        #将msg1转为dataframe
        self.Key_composition = pd.DataFrame([msg1],index=['formula_dict_keys'])

        self.Train_composition = alt.Chart(msg2).mark_arc(innerRadius=50,outerRadius=100).encode(
            theta="count",
            color="is_train:N",
            tooltip=['count']
        )

        #绘制饼形图，title为类别构成
        self.Class_composition  = alt.Chart(msg3).mark_arc(innerRadius=50,outerRadius=100).encode(
            theta="count", 
            color="group:N",
            tooltip=['group','count']
        )

        #将data中formula_mass列的值用altair画图
        #图例在左边
        self.Mass_composition = alt.Chart(msg4).mark_bar().encode(
            x=alt.X('formula_mass',bin=alt.Bin(maxbins=50)),
            y='count()',
            tooltip=['count()'])


if __name__ == '__main__':

    a= Dataset_vis()
    st.title('数据集统计')
    st.subheader('数据集中的元素构成')
    st.write(a.Key_composition)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('数据集中的可训练的数量')
        st.write(a.Train_composition,use_container_width=True)
        st.subheader('数据集中的mass分布')
        st.write(a.Mass_composition,use_container_width=True)

    with col2:
        st.subheader('数据集中的group构成')
        st.write(a.Class_composition,use_container_width=True)
    
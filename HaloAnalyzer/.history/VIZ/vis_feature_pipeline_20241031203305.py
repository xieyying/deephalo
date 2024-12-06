import streamlit as st
import pandas as pd
import altair as alt
import ast
import os

class PipelineVis:
    def __init__(self, result_folder):
        self.result_folder = result_folder

    def load_base_data(self, file_path):
        # 读取数据
        self.scan = pd.read_csv(file_path)
        self.scan['masstrace_centroid_mz'] = self.scan['masstrace_centroid_mz'].apply(lambda x: list(ast.literal_eval(x)))
        self.scan['masstrace_intensity'] = self.scan['masstrace_intensity'].apply(lambda x: list(ast.literal_eval(x)))
        self.scan['RT'] = self.scan['RT']/60
        # Step 2: Convert 'inty_list' column values from tuples to lists
        # 展开mz_list和inty_list
        rows = []
        for _, row in self.scan.iterrows():
            for mz, inty in zip(row["masstrace_centroid_mz"], row["masstrace_intensity"]):
                rows.append({"RT": row["RT"], "mz": mz, "inty": inty, "class_pred": row["class_pred"], 'charge': row['charge'],'m2_m1': row['m2_m1'],\
                   'm1_m0': row['m1_m0'],'p0_int':row['p0_int'],'p1_int':row['p1_int'],'p2_int':row['p2_int'],'p3_int':row['p3_int'],'p4_int':row['p4_int'],\
                       'p5_int':row['p5_int'],'p6_int':row['p6_int'],'reconstruction_error':row['reconstruction_error'],'H_score':row['H_score']})  # 根据需要从self
        expanded_df = pd.DataFrame(rows)
        
        # Ensure 'inty' column is numeric
        expanded_df['inty'] = pd.to_numeric(expanded_df['inty'], errors='coerce')

        x_scale = alt.Scale(domain=(self.scan['RT'].min(), self.scan['RT'].max()))
        selection = alt.selection_point(fields=['class'], bind='legend')
        # 将self.target中的intensity列转换为0-1之间的数
        # expanded_df['inty'] = expanded_df['inty'] / expanded_df['inty'].max()

        # 绘制点图
        base = alt.Chart(expanded_df)
        color_scale = alt.Scale(
            domain=list(range(8)),  # classes 0-7
            range=['green', 'blue', 'indigo', 'violet', 'red', 'orange', 'yellow', 'black']  # corresponding colors
        )
        self.target_chart = base.mark_point().encode(
            alt.X('RT', scale=x_scale),
            y='mz',
            color=alt.Color('class_pred:N', scale=color_scale),
            size='inty',
            tooltip=['RT', 'mz', 'inty', 'class_pred', 'charge', 'm2_m1', 'm1_m0', 'p0_int', 'p1_int', 'p2_int', 'p3_int', 'p4_int', 'p5_int', 'p6_int','reconstruction_error','H_score'],
            opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
        ).interactive().add_params(
            selection
        ).properties(
            width=1000,  # Set the width to 1000 pixels
            height=600  # Set the height to 600 pixels
        )
        return self.target_chart

    def main(self):
        # 读取result_folder目录下所有以_scan.csv结尾的文件
        files = os.listdir(self.result_folder)
        files = [file for file in files if file.endswith('feature.csv')]
        file_paths = [os.path.join(self.result_folder, file) for file in files]
        charts = [self.load_base_data(file_path) for file_path in file_paths]
        # 绘制图表
        for i in range(len(charts)):
            st.write(files[i])
            st.write(charts[i])

if __name__ == '__main__':
    project_path = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms_noClFe\2M_fake_molecules\result\halo'
    # project_path = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms_noClFe\2M_fake_molecules\result\Fe'
    # project_path = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms_noClFe\2M_fake_molecules\result\cmx_p9'
    # pv = PipelineVis(project_path)
    # pv.main()
    
    files = os.listdir(project_path)
    files = [file for file in files if file.endswith('feature.csv')]
    output = project_path + '\\RT_less_9'
    for f in files:
        df = pd.read_csv(f)
        df = df[df['RT']<540]
        if df.shape[0] > 0:
            df.to_csv(output + '\\' + f, index=False)
import streamlit as st
import pandas as pd
import altair as alt
import ast
import os

class PipelineVis:
    def __init__(self, result_folder):
        self.result_folder = result_folder

    def load_base_data(self, file_path, file_name):
        # 读取数据
        scan = pd.read_csv(file_path)
        scan['masstrace_centroid_mz'] = scan['masstrace_centroid_mz'].apply(lambda x: list(ast.literal_eval(x)))
        scan['masstrace_intensity'] = scan['masstrace_intensity'].apply(lambda x: list(ast.literal_eval(x)))
        scan['RT'] = scan['RT'] / 60
        # 展开mz_list和inty_list
        rows = []
        for _, row in scan.iterrows():
            for mz, inty in zip(row["masstrace_centroid_mz"], row["masstrace_intensity"]):
                rows.append({"RT": row["RT"], "mz": mz, "inty": inty, "class_pred": row["class_pred"], 'charge': row['charge'], 'm2_m1': row['m2_m1'],
                             'm1_m0': row['m1_m0'], 'p0_int': row['p0_int'], 'p1_int': row['p1_int'], 'p2_int': row['p2_int'], 'p3_int': row['p3_int'], 'p4_int': row['p4_int'],
                             'p5_int': row['p5_int'], 'p6_int': row['p6_int'], 'reconstruction_error': row['reconstruction_error'], 'H_score': row['H_score'], 'file': file_name})
        expanded_df = pd.DataFrame(rows)
        
        # Ensure 'inty' column is numeric
        expanded_df['inty'] = pd.to_numeric(expanded_df['inty'], errors='coerce')
        
        return expanded_df

    def main(self):
        # 读取result_folder目录下所有以.csv结尾的文件
        files = os.listdir(self.result_folder)
        files = [file for file in files if file.endswith('.csv')]
        file_paths = [os.path.join(self.result_folder, file) for file in files]

        # 合并所有文件的数据
        all_data = pd.concat([self.load_base_data(file_path, file) for file_path, file in zip(file_paths, files)], ignore_index=True)

        # 绘制柱状图
        base = alt.Chart(all_data)
        color_scale = alt.Scale(
            domain=list(range(8)),  # classes 0-7
            range=['green', 'blue', 'indigo', 'violet', 'red', 'orange', 'yellow', 'black']  # corresponding colors
        )
        bar_chart = base.mark_bar().encode(
            x=alt.X('file:N', title='File'),
            y=alt.Y('sum(inty):Q', title='Total Intensity'),
            color=alt.Color('class_pred:N', scale=color_scale),
            tooltip=['file', 'sum(inty)', 'class_pred']
        ).properties(
            width=1000,  # Set the width to 1000 pixels
            height=600  # Set the height to 600 pixels
        ).interactive()

        st.altair_chart(bar_chart)

if __name__ == '__main__':
    project_path = r'C:\Users\xyy\Desktop\python\HaloAnalyzer_training\022_six_dataset_openms_noClFe\result\F1_plus_standards\target_compounds\target_rt_mz_feature'
    pv = PipelineVis(project_path)
    pv.main()
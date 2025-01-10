import os
import shutil

# 文件列表
files_to_copy = [
    r'K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_030_hv_M6_cmx_p9_E6_nr1.mzML',
    r'K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_10_20_M3_cmx_p9_D3_nr1.mzML',
    r'K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_11_23_M11_cmx_p9_C11_nr1.mzML',
    r'K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_11_39_M11_cmx_p9_G11_nr1.mzML',
    r'K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_8_7_M11_cmx_p9_B11_nr1.mzML',
    r'K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_8_7_M2_cmx_p9_B2_nr1.mzML',
    r'K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_8_7_M3_cmx_p9_B3_nr1.mzML',
    r'K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_8_7_M5_cmx_p9_B5_nr1.mzML',
    r'K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\WJH_OSMAC_P7\Streptomyces_sp001298545_XZ_20_671_M11_WJH_OSMAC_P7_F11.mzML',
    r'K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\WJH_OSMAC_P7\Streptomyces_sp_XZ_19_359_M11_WJH_OSMAC_P7_B11.mzML',
    r'K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\WJH_OSMAC_P7\Streptomyces_sp_XZ_19_359_M8_WJH_OSMAC_P7_B8.mzML'
]

# 目标文件夹
destination_folder = r'K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\error'

# 如果目标文件夹不存在，则创建它
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 复制文件
for file in files_to_copy:
    if os.path.exists(file):
        shutil.copy(file, destination_folder)
    else:
        print(f"File not found: {file}")
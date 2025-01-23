import shutil
import os

# List of file paths
file_paths = [
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p15\Strepomyces_cmx_5_11_M2_cmx_p15_A2_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p15\Strepomyces_067_1_M9_cmx_p15_E9_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p15\Strepomyces_cmx_5_11_M2_cmx_p15_A2_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p15\Strepomyces_cmx_5_11_M8_cmx_p15_A8_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p16_mzml\Strepomyces_cmx_13_2_M9_cmx_p16_A9_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p16_mzml\Strepomyces_cmx_4_9_M10_cmx_p16_B10_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p16_mzml\Strepomyces_cmx_5_10_M11_cmx_p16_E11_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p16_mzml\Strepomyces_cmx_5_10_M6_cmx_p16_E6_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p17\Strepomyces_029_5_M11_cmx_p17_D11_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p15\Strepomyces_067_1_M9_cmx_p15_E9_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p15\Strepomyces_cmx_5_11_M2_cmx_p15_A2_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p15\Strepomyces_cmx_5_11_M8_cmx_p15_A8_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p16_mzml\Strepomyces_cmx_13_2_M9_cmx_p16_A9_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p16_mzml\Strepomyces_cmx_4_9_M10_cmx_p16_B10_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p16_mzml\Strepomyces_cmx_5_10_M11_cmx_p16_E11_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p16_mzml\Strepomyces_cmx_5_10_M6_cmx_p16_E6_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p17\Strepomyces_021_4_M7_cmx_p17_F7_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p17\Strepomyces_029_5_M11_cmx_p17_D11_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p17\Strepomyces_cmx_10_37_M3_cmx_p17_H3_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_p5_mzML\Strepomyces_cmx_6_16_M9_cmx_p5_C9_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_030_hv_M12_cmx_p9_E12_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_030_hv_M6_cmx_p9_E6_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_10_19_M5_cmx_p9_A5_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_10_19_M7_cmx_p9_A7_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_10_19_M8_cmx_p9_A8_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_10_20_M3_cmx_p9_D3_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_10_20_M5_cmx_p9_D5_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_10_20_M7_cmx_p9_D7_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_11_23_M11_cmx_p9_C11_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_11_39_M11_cmx_p9_G11_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_11_39_M5_cmx_p9_G5_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_11_39_M6_cmx_p9_G6_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_11_39_M7_cmx_p9_G7_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_11_39_M9_cmx_p9_G9_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_8_7_M11_cmx_p9_B11_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_8_7_M12_cmx_p9_B12_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_8_7_M2_cmx_p9_B2_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_8_7_M3_cmx_p9_B3_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_8_7_M5_cmx_p9_B5_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_8_7_M6_cmx_p9_B6_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_8_7_M7_cmx_p9_B7_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_8_7_M8_cmx_p9_B8_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\cmx_P9_mzML\Strepomyces_cmx_8_7_M9_cmx_p9_B9_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\error\Strepomyces_030_hv_M6_cmx_p9_E6_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\error\Strepomyces_cmx_10_20_M3_cmx_p9_D3_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\error\Strepomyces_cmx_11_23_M11_cmx_p9_C11_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\error\Strepomyces_cmx_11_39_M11_cmx_p9_G11_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\error\Strepomyces_cmx_8_7_M11_cmx_p9_B11_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\error\Strepomyces_cmx_8_7_M2_cmx_p9_B2_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\error\Strepomyces_cmx_8_7_M3_cmx_p9_B3_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\error\Strepomyces_cmx_8_7_M5_cmx_p9_B5_nr1.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\error\Streptomyces_sp001298545_XZ_20_671_M11_WJH_OSMAC_P7_F11.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\error\Streptomyces_sp_XZ_19_359_M11_WJH_OSMAC_P7_B11.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\error\Streptomyces_sp_XZ_19_359_M8_WJH_OSMAC_P7_B8.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\WJH_OSMAC_P7\Streptomyces_sp001298545_XZ_20_671_M11_WJH_OSMAC_P7_F11.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\WJH_OSMAC_P7\Streptomyces_sp_XZ_19_359_M11_WJH_OSMAC_P7_B11.mzML",
    r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\WJH_OSMAC_P7\Streptomyces_sp_XZ_19_359_M8_WJH_OSMAC_P7_B8.mzML"
]

# Destination folder
destination_folder = r"K:\chenmingxu\CMX_OSMAC_raw_data_compressed\two_parts\P5_P9_P15_P19\error"

# Copy files to the destination folder
for file_path in file_paths:
    try:
        shutil.copy(file_path, destination_folder)
        # print(f"Copied: {file_path}")
    except Exception as e:
        print(f"Failed to copy {file_path}: {e}")
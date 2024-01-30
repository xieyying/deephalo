from HaloAnalyzer.main import *
import os
import argparse
from .parameters import run_parameters
import importlib_resources
from .model_test import timeit
import logging
#通过终端选择运行模式
__version__ = '0.2.0'

@timeit
def main():

    print("\n\nHaloAnalyzer (%s) \n" %__version__)
    #命令行参数设置
    parser = argparse.ArgumentParser(description='HALOAnalyzer: a tool for mining halogenates based on high resolution mass data.')

    parser.add_argument('-v', '--version', action='version', version=__version__, 
            help='print version and exit')
    parser.add_argument('run', metavar='subcommand', 
            help='one of the subcommands: create_dataset, train_model, analyze_mzml, viz_result')
    parser.add_argument('-i', '--input', 
            help='input directory of mzML files to process, or a single file to analyze')
    parser.add_argument('-b', '--blank', 
            help='input directory of blank mzML files for substraction')
    parser.add_argument('-p', '--project', 
            help='project path')
    parser.add_argument('-m', '--mode', 
            help='train model mode: manual or search')
    parser.add_argument('-l', '--list_rois',  nargs='+', type=int,
            help='list of rois to extract ms2 spectra')
    parser.add_argument('-ob', '--overwrite_blank', action='store_true',
                    help='overwrite the original blank output files')
    args = parser.parse_args()

    #处理命令行参数
    if args.run not in ['create_dataset', 'train_model', 'analyze_mzml', 'viz_result','extract_ms2']:
        print("Expecting one of the subcommands: create_dataset, train_model, analyze_mzml, viz_result, extract_ms2.")
    else:
        if args.project != None:
            os.chdir(args.project)

            if args.run == 'create_dataset':
                pipeline_dataset()
            elif args.run == 'train_model':
                pipeline_model(args.mode)
            elif args.run == 'analyze_mzml':
                blank_path = args.blank
                mzml_path = args.input
                if blank_path != None:
                    #如果存在./test_mzml_prediction/blank/merged_blank_halo.csv则不再进行blank_analyze_mzml
                    if not os.path.exists('./test_mzml_prediction/blank/merged_blank_halo.csv') or args.overwrite_blank:
                        blank_analyze_mzml(blank_path)
                    if os.path.isdir(mzml_path):
                        batch_find_halo_substrate_blank(mzml_path)
                    elif os.path.isfile(mzml_path):
                        pipeline_find_halo_substrate_blank(mzml_path)
                elif os.path.isdir(mzml_path):   
                    batch_find_halo_no_blank(mzml_path)
                elif os.path.isfile(mzml_path):
                    pipeline_find_halo_no_blank(mzml_path)
                else:
                    print("Please specify a mzML file or a folder containing mzML files to analyze.")
                # with open(r'test_mzml_prediction/log.txt','w') as f: f.write(mzml_path)

            elif args.run == 'viz_result':

                #更新config中的vis_path
                parameters = run_parameters()
                c = parameters.config
                c['visualization']['path'] = args.project
                parameters.update(c)
                # 运行vis.py
                vis_path = importlib_resources.files('HaloAnalyzer') / 'vis.py'
                print(vis_path)
                os.system('python -m streamlit run %s' %vis_path)
            elif args.run == 'extract_ms2':
                mzml_path = args.input
                project_path = args.project
                rois_list = args.list_rois
                if mzml_path == None:
                    print("Please specify a mzML file to analyze.")
                if rois_list == None:
                    print("Please specify a list of rois to extract ms2 spectra.")
                if project_path == None:
                    print("Please specify a project path.")
                if mzml_path != None and rois_list != None and project_path != None:
                    pipeline_extract_ms2_of_rois(mzml_path,project_path,rois_list)
                
        else:
            print("Please specify a project path.")

  
if __name__ == '__main__':
    main()
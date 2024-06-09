from HaloAnalyzer.main import pipeline_dataset,pipeline_model,pipeline_analyze_mzml
import os
import argparse
from .parameters import run_parameters
import importlib_resources
from .model_test import timeit
import logging


#通过终端选择运行模式
__version__ = '0.9'

@timeit
def main():
    """
    Main function to run the HaloAnalyzer from the command line.
    """
    #打印版本信息
    print("\n\nHaloAnalyzer (%s) \n" %__version__)

    #命令行参数设置
    parser = argparse.ArgumentParser(description='HALOAnalyzer: a tool for mining halogenates based on high resolution mass data.')
    parser.add_argument('-v', '--version', action='version', version=__version__, help='print version and exit')
    parser.add_argument('run', metavar='subcommand', help='one of the subcommands: create_dataset, train_model, analyze_mzml, viz_result')
    parser.add_argument('-i', '--input', help='input directory of mzML files to process, or a single file to analyze')
    parser.add_argument('-b', '--blank', help='input directory of blank mzML files for substraction')
    parser.add_argument('-p', '--project', help='set the project path for HaloAnalyzer output')
    parser.add_argument('-m', '--mode', help='train model mode: manual or search')
    parser.add_argument('-l', '--list_rois',  nargs='+', type=int, help='list of rois to extract ms2 spectra')
    parser.add_argument('-ob', '--overwrite_blank', action='store_true', help='overwrite the original blank output files')
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
                print(args.mode)
                pipeline_model(args.mode)
            elif args.run == 'analyze_mzml':
                pipeline_analyze_mzml(args)
            elif args.run == 'viz_result':
                # pipeline_viz_result()
                pass
            
        else:
            print("Please specify a project path.")

  
if __name__ == '__main__':
    main()
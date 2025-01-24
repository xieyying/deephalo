import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from DeepHalo.main import pipeline_dataset,pipeline_model,pipeline_analyze_mzml,pipeline_dereplication
from .parameters import RunParameters
from .model_test import timeit
import typer

#通过终端选择运行模式
__version__ = '0.9'

para = RunParameters()
app = typer.Typer()

@app.command()
def create_dataset(
    project_path: str = typer.Argument(..., help="The path of the project"),
    ):
    """
    Create a Dataset object and execute the workflow
    """
    os.chdir(project_path)
    pipeline_dataset(para)

@app.command()
def create_model(
    project_path: str = typer.Argument(..., help="The path of the project"),
    mode: str = typer.Option('manual', '-m', '--mode', help="The mode of the training process: manual or search"),
    ):
    """
    Create a Model object and execute the workflow
    """
    os.chdir(project_path)
    para.args_mode = mode
    pipeline_model(para)

@app.command()
def analyze_mzml(
    project_path: str = typer.Argument(..., help="The path of the project"),
    input_path: str = typer.Argument(..., help="The path of the input mzML file"),
    blank_path: str = typer.Option(None, '-b', '--blank', help='input directory of blank mzML files for subtraction'),
    overwrite_blank: bool = typer.Option(False, '-ob', '--overwrite-blank', help='overwrite blank mode: True or False'),
    ms2: bool = typer.Option(False, '-ms2', '--ms2', help='ms2 mode: True or False'),
    ):
    """
    Analyze mzML files
    """
    os.chdir(project_path)
    para.args_input = input_path
    para.args_blank = blank_path
    para.args_overwrite_blank = overwrite_blank
    para.args_ms2 = ms2
    pipeline_analyze_mzml(para)

@app.command()
def dereplication(
    project_path: str = typer.Argument(..., help="The path of the project"),
    GNPS_folder: str = typer.Option(None, '-g', '--GNPS-file', help='The path of the GNPS file'),
    user_database: str = typer.Option(None, '-ud', '--user-database', help='The path of the user database and key of the column'),
    user_database_key: str = typer.Option(None, '-udk', '--user-database-key', help='The key of the column in the user database'),
    ):
    """
    Dereplication
    """
    if GNPS_folder == None and user_database == None:
        raise ValueError('Please provide the GNPS file or user database')
    para.args_project_path = project_path
    para.args_GNPS_folder = GNPS_folder
    para.args_user_database = user_database
    para.args_user_database_key = user_database_key
    pipeline_dereplication(para)

if __name__ == '__main__':
    app()
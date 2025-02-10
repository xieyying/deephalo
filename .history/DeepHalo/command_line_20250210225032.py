import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from DeepHalo.main import pipeline_dataset, pipeline_model, pipeline_analyze_mzml, pipeline_dereplication
from .parameters import RunParameters
from .model_test import path_check
import typer
import time


# CLI interface for DeepHalo with version information
__version__ = '0.9'

app = typer.Typer()

def timer_decorator(func):
    """Decorator to measure and display function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        typer.echo(f"\nCommand execution time: {elapsed_time:.2f} seconds")
        return result
    return wrapper

@timer_decorator
@app.command()
def analyze_mzml(
    input_path: str = typer.Option(
        ...,
        '-i', '--input',
        help="Input path (single .mzML file or directory containing multiple .mzML files)"),
    project_path: str = typer.Option(
        ...,
        '-o', '--output',
        help="Output directory path for project files"),
    user_config: str = typer.Option(
        None,
        '-c', '--config',
        help="[Optional] Custom configuration file path to override defaults"),
    blank_path: str = typer.Option(
        None,
        '-b', '--blank',
        help="[Optional] Directory containing blank sample mzML files for subtraction"),
    overwrite_blank: bool = typer.Option(
        False,
        '-ob', '--overwrite-blank',
        help="[Optional] Force regenerate blank sample results (default: False)"),
    ms2: bool = typer.Option(
        False,
        '-ms2', '--ms2',
        help="[Optional] Enable MS2 data extraction for detected features (default: False)")
):
    """High-throughput detection of halogenated compounds"""
    # Create output directory if it doesn't exist
    path_check(project_path)
    
    os.chdir(project_path)
    para = RunParameters(user_config=user_config)
    para.args_input = input_path
    para.args_blank = blank_path
    para.args_overwrite_blank = overwrite_blank
    para.args_ms2 = ms2
    pipeline_analyze_mzml(para)

@app.command()
@timer_decorator
def dereplication(
    project_path: str = typer.Option(
        ...,
        '-o', '--output',
        help="Output directory path for project files"),
    user_config: str = typer.Option(
        None,
        '-c', '--config',
        help="[Optional] Custom configuration file path to override defaults"),
    GNPS_folder: str = typer.Option(
        None,
        '-g', '--GNPS-file',
        help="Path to GNPS output folder"),
    user_database: str = typer.Option(
        None,
        '-ud', '--user-database',
        help="Path to user database file in .csv or .json format"),
    user_database_key: str = typer.Option(
        None,
        '-udk', '--user-database-key',
        help="Database key specifying formula column for dereplication")
):
    """Perform dereplication using GNPS output and/or user database"""
    
    if GNPS_folder is None and user_database is None:
        raise ValueError('Missing required GNPS file or user database')
    
    para = RunParameters(user_config=user_config)
    para.args_project_path = project_path
    para.args_GNPS_folder = GNPS_folder
    para.args_user_database = user_database
    para.args_user_database_key = user_database_key
    pipeline_dereplication(para)

@app.command()
@timer_decorator
def create_dataset(
    project_path: str = typer.Argument(
        ...,
        help="Project directory path for dataset creation"),
    user_config: str = typer.Option(
        None,
        '-c', '--config',
        help="[Optional] Custom configuration file path to override defaults")
):
    """Create Dataset object and execute workflow"""
    para = RunParameters(user_config=user_config)
    os.chdir(project_path)
    pipeline_dataset(para)

@app.command()
@timer_decorator
def create_model(
    project_path: str = typer.Argument(
        ...,
        help="Project directory path for model creation"),
    user_config: str = typer.Option(
        None,
        '-c', '--config',
        help="[Optional] Custom configuration file path to override defaults"),
    mode: str = typer.Option(
        'manual',
        '-m', '--mode',
        help="Training mode: 'manual' or 'search' (default: 'manual')")
):
    """Create Model object and execute workflow"""
    os.chdir(project_path)
    para = RunParameters(user_config=user_config)
    para.args_mode = mode
    pipeline_model(para)

if __name__ == '__main__':
    app()
""" 
Loads necessary paths to work with from scripts/ and notebooks/ directories.
Uses ../folder/ structure. Thus make sure that script is called from the dir **one** level down.
"""
from pathlib import Path

path_data = Path("../data/")
path_figures = Path("../figures/")
path_scripts = Path("../scripts/")
path_notebooks = Path("../notebooks")
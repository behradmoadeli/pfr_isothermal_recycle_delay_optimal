from .set_directory import set_directory
from .obtain_default_pars import obtain_default_pars
from .create_custom_pars_list import create_custom_pars_list
from .load_dataframe_from_csv import load_dataframe_from_csv
from .generate_unique_file_path import generate_unique_file_path
from .significant_digits import significant_digits
from .process_dataframe import process_dataframe
from .char_eq import char_eq
from .my_fsolve import my_fsolve
from .find_eig import find_eig
from .save_dataframe_to_csv import save_dataframe_to_csv
from .create_label import create_label
from .plot_single_df import plot_single_df
from sys import version_info
from .char_eq_adjoint import char_eq_adjoint
from .my_fsolve_adjoint import my_fsolve_adjoint
from .find_eig_adjoint import find_eig_adjoint


# Check if Python version is 3.11 or higher
if version_info >= (3, 11):
    from .plot_multiple_datasets import plot_multiple_datasets
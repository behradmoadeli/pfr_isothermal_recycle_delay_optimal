import os
from functions import *

# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script's directory
os.chdir(script_directory)

# Now the CWD is the same as the script's directory
print("New Current Working Directory:", os.getcwd())


pars_list = create_custom_pars_list('pars_list.csv')
r = [-100, 20, 45]
i = [0, 360, 120]
f = [(-100, 25), 300]

for par in pars_list:
    df, label, metadata = find_eig(par=par, guess_range_real=r, guess_range_imag=i)
    save_dataframe_to_csv(df, label, 'CSV', metadata)

plot_multiple_datasets('CSV', save_path='multi_plot', rows=2, filter=True, filter_range=f)
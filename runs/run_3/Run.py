import os
from functions import *

# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script's directory
os.chdir(script_directory)

# Now the CWD is the same as the script's directory
print("New Current Working Directory:", os.getcwd())

guess = {
    'guess_range_real':[-10, -5, 6],
    'guess_range_imag':[0, 10, 6]
}
f = [(-50, 10), 50]

pars_list, default_pars = create_custom_pars_list('pars_list.csv')

df, label, metadata = find_eig(default_pars, **guess)
save_dataframe_to_csv(df, label, 'CSV', metadata)

for par in pars_list:
    df, label, metadata = find_eig(par, **guess)
    save_dataframe_to_csv(df, label, 'CSV', metadata)

plot_multiple_datasets('CSV', save_path='multi_plot', rows=3, filter=True, filter_range=f)
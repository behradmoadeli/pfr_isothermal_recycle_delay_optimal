import os
from functions import *

# Set working directory to current location of the script
working_directory = set_directory(__file__)

# Prepare kwargs for find_eig() funcition
guess = {
    'guess_range_real':[-275,24,150],
    'guess_range_imag':[0,100,50]
}

# Get default_pars and their coresponding solution
# default_pars = obtain_default_pars('pars_list.csv')
# print("Working on default set")
# save_dataframe_to_csv(*find_eig(default_pars, **guess, round_sig_digits=3), 'CSV')

# Get the solutions for parameter variation
# pars_list = create_custom_pars_list('pars_list.csv')
# for par in pars_list:
#     print("Working on a new custom set")
#     save_dataframe_to_csv(*find_eig(par, default_pars=default_pars, **guess), 'CSV')

df = plot_single_df('CSV/default.csv')[0]
print(df)

f = [(-200, 10), 300]
plot_multiple_datasets('CSV', save_path='multi_plot', rows=2, filter=True, filter_range=f)
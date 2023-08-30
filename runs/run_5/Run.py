import os
from functions import *

# Set working directory to current location of the script
working_directory = set_directory(__file__)

# Prepare kwargs for find_eig() funcition
guess = {
    'guess_range_real':[-50,10,30],
    'guess_range_imag':[200,300,20]
}
f = [(-50, 10), 50]

default_pars = obtain_default_pars('pars_list.csv')
save_dataframe_to_csv(*find_eig(default_pars, **guess, round_sig_digits=3), 'CSV')

# df = plot_single_df('CSV/default_4.csv')[0]
# print(df)

pars_list = create_custom_pars_list('pars_list.csv')
for par in pars_list:
    save_dataframe_to_csv(*find_eig(par, default_pars=default_pars, **guess), 'CSV')

# plot_multiple_datasets('CSV', save_path='multi_plot', rows=3, filter=False)
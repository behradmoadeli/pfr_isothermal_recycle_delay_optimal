import os
from functions import *

# Set working directory to current location of the script
working_directory = set_directory(__file__)

# Prepare kwargs for find_eig() funcition
guess = {
    'guess_range_real':[-50,5,110],
    'guess_range_imag':[0,1200,1200]
}

popt = [38.062, -7.61376e-02, -37.225]
offset_abs_tol = 5
offset_rel_tol = 0.15
mesh_filter = [*popt, offset_abs_tol, offset_rel_tol]

# Get default_pars and their coresponding solution
default_pars = obtain_default_pars('pars_list.csv')
print("Working on default set")

save_dataframe_to_csv(*find_eig(default_pars, **guess, mesh_filter=mesh_filter, round_sig_digits=4, tol_is_sol=1e-6, max_iter=200), 'CSV')

df = plot_single_df('CSV/default.csv')[0]
print(df)

# f = [(-100, 10), 100]
# plot_multiple_datasets('CSV', save_path='multi_plot', rows=1, filter=True, filter_range=f)
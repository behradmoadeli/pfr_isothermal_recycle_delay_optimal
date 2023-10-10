def find_eig_adjoint(init_guess_df, par=None, default_pars=None, **kwargs):

    import numpy as np
    import pandas as pd
    import scipy.optimize as opt
    from .char_eq_adjoint import char_eq_adjoint
    from .create_label import create_label
    from .obtain_default_pars import obtain_default_pars
    from .my_fsolve_adjoint import my_fsolve_adjoint

    import warnings
    warnings.simplefilter('ignore')


    # Create appropriate label for par
    if par['label'] == 'default':
        default_pars = par.copy()

    if not default_pars:
        default_pars = obtain_default_pars(kwargs.get('pars_list_path', 'pars_list.csv'))
    
    if par['label'] == '':
        par['label'] = create_label(par, default_pars=default_pars, **kwargs)

    par['label'] += '_adj'
    
    # Assign default values to the rest of missing keyword arguments
    tol_fsolve = kwargs.get('tol_fsolve', 1e-9)
    tol_is_sol = kwargs.get('tol_is_sol', 5e-3)
    round_sig_digits = kwargs.get('round_sig_digits', 3)

    metadata = {
        'par': par,
        'tols': (tol_fsolve, tol_is_sol, round_sig_digits)
    }

    # Constructiong a dictionary to capture legit solutions
    solution_dict = {
        'Sol_r':[], 'Sol_i':[], 'Guess':[], 'g(x)':[], 'ier':[], 'msg':[], 'infodict':[]
    }

    init_guess_complex = init_guess_df['Sol_r'] + 1j * init_guess_df['Sol_i']

    for m in init_guess_complex:
        # print(f"Getting results for guess = {m.real:.2f} + {m.imag:.2f}j...")
        m = np.array([m.real, m.imag])
        solution_dict = my_fsolve_adjoint(char_eq_adjoint, m, par, tol_fsolve, tol_is_sol, 25, solution_dict, full_output=True)
    solution_df = pd.DataFrame(solution_dict)
    return (solution_df, par['label'], metadata)
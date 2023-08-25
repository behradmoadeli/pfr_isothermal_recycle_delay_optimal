def find_eig(**kwargs):
    """
    This function solves the char equation in complex plane for different initial guesses.

    Parameters:
        **kwargs (keyword arguments):            
            - par (dict): A dictionary containing parameters for the system's matrix. If not provided, keys may be passed separately. Absent keys will take default values.
            - guess_single (complex): A single initial guess for eigenvalue calculation (real + imaginary part).
            - guess_range_real (list): A list specifying the range of real parts of initial guess values.
            - guess_range_imag (list): A list specifying the range of imaginary parts of initial guess values.
            - tol_fsolve (float): Tolerance for fsolve array-like comaprison to converge.
            - tol_is_sol (float): Tolerance for a complex solution to be accepted.
            - round_sig_digits (float): Number of significant digits to either separate two different solutions or merge them as one.

    Returns:
        tuple:
            A tuple containing:

            - solution_df (pandas.DataFrame): DataFrame containing found solutions' information.
            - label (str): A label describing the customized parameters used for the computation.
            - metadata (dict): A dictionary containing input parameter values used in the computation.
    """

    import numpy as np
    import pandas as pd
    import scipy.optimize as opt
    from .char_eq import char_eq
    from .process_dataframe import process_dataframe

    import warnings
    warnings.simplefilter('ignore')
    
    default_pars = {
        'k': 10,
        'D': 0.1,
        'v': 0.5,
        'tau': 1,
        'R': 0.9,
        'label': 'default'
    }

    # Assign default values to missing keyword arguments for parameters
    label_needed = False
    if 'par' in kwargs:
        par = kwargs['par']
        if par['label'] == '':
            label_needed = True
    else:
        par = default_pars.copy()
        for key in par:
            par[key] = kwargs.get(key, par[key])
        if par != default_pars:
            label_needed = True
    # Creating a label for parameters if needed
    if label_needed:
        # default_pars[key], par['label'] = 0 , 0
        # differing_pairs = {key: value for key, value in par.items() if not np.isclose(value, default_pars[key])}
        differing_pairs = {}
        for key, value in par.items():
            if key != 'label':
                if not np.isclose(value, default_pars[key]):
                    differing_pairs[key] = value
        par['label'] = '_'.join(
            [f"({key}_{value:.3g})" for key, value in differing_pairs.items()])

    # Assign default values to missing keyword arguments for initial guess values
    if 'guess_single' in kwargs:
        guess_single_r = np.real(kwargs['guess_single'])
        guess_single_i = np.imag(kwargs['guess_single'])

        guess_range_real = [guess_single_r, guess_single_r, 1]
        guess_range_imag = [guess_single_i, guess_single_i, 1]
    else:
        guess_range_real = kwargs.get('guess_range_real', [-300, 50, 350])
        guess_range_imag = kwargs.get('guess_range_imag', [0, 200, 200])

    # Assign default values to the rest of missing keyword arguments
    tol_fsolve = kwargs.get('tol_fsolve', 1e-15)
    tol_is_sol = kwargs.get('tol_is_sol', 1e-6)
    round_sig_digits = kwargs.get('round_sig_digits', 4)

    metadata = {
        'par': par,
        'guess_range': (guess_range_real, guess_range_imag),
        'tols': (tol_fsolve, tol_is_sol, round_sig_digits)
    }

    # Constructiong a dictionary to capture legit solutions
    solution_dict = {
        'Sol_r': [], 'Sol_i': [], 'Guess_r': [], 'Guess_i': [], 'g(x)': []
    }

    # Constructiong a 2D (Re-Im plane) mesh for different initial guess values
    mesh_builder = np.meshgrid(
        np.linspace(guess_range_real[0],
                    guess_range_real[1], guess_range_real[2]),
        np.linspace(guess_range_imag[0],
                    guess_range_imag[1], guess_range_imag[2])
    )
    mesh = mesh_builder[0] + mesh_builder[1] * 1j

    for i in mesh:
        for m in i:
            # obtaining an initial guess from the mesh as a complex number
            m = np.array([m.real, m.imag])
            solution_array = opt.fsolve(char_eq, m, par, xtol=tol_fsolve)
            # evaluationg the value of char_eq at the obtained relaxed solution
            is_sol = char_eq(solution_array, par)
            is_sol = abs(complex(is_sol[0], is_sol[1]))
            if np.isclose(is_sol, 0, atol=tol_is_sol):
                solution_dict['Guess_r'].append(m[0])
                solution_dict['Guess_i'].append(m[1])
                solution_dict['g(x)'].append(is_sol)
                solution_dict['Sol_r'].append(solution_array[0])
                solution_dict['Sol_i'].append(solution_array[1])
                solution_array_conj_guess = solution_array.copy()
                solution_array_conj_guess[1] *= -1
                solution_array_conj = opt.fsolve(
                    char_eq, solution_array_conj_guess, par, xtol=tol_fsolve)
                # evaluationg the value of char_eq at the obtained relaxed solution
                is_sol_conj = char_eq(solution_array_conj, par)
                is_sol_conj = (abs(complex(is_sol_conj[0], is_sol_conj[1])))
                if np.isclose(is_sol_conj, 0, atol=tol_is_sol):
                    solution_dict['Guess_r'].append(m[0])
                    solution_dict['Guess_i'].append(-m[1])
                    solution_dict['g(x)'].append(is_sol_conj)
                    solution_dict['Sol_r'].append(solution_array_conj[0])
                    solution_dict['Sol_i'].append(solution_array_conj[1])

    solution_df = process_dataframe(
        pd.DataFrame(solution_dict), round_sig_digits)
    solution_df = solution_df.sort_values(by=['Sol_r'], ascending=False)
    return (solution_df, par['label'], metadata)
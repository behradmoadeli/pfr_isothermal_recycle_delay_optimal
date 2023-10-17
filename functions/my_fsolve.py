def my_fsolve(my_fun, m, par, tol_fsolve, tol_is_sol, max_iterations, solution_dict, full_output=True):
    import numpy as np
    import pandas as pd
    import scipy.optimize as opt
    from .create_label import create_label
    from .obtain_default_pars import obtain_default_pars

    temp_guess = m
    sol = False
    sol_conj = False
    for _ in range(max_iterations):
        solution_array, infodict, ier, msg  = opt.fsolve(
            my_fun, temp_guess, par, xtol=tol_fsolve, full_output=full_output)
        is_sol = abs(complex(*my_fun(solution_array, par)))
        if np.isclose(is_sol, 0, atol=tol_is_sol):
            sol = True
            solution_array, infodict, ier, msg = opt.fsolve(
                my_fun, solution_array, par, xtol=10*tol_is_sol, full_output=full_output)
            is_sol = abs(complex(*my_fun(solution_array, par)))
            if is_sol < tol_is_sol:
                solution_array_conj_guess = solution_array.copy()
                solution_array_conj_guess[1] *= -1
                for _ in range(int(np.ceil(np.sqrt(max_iterations)))):
                    solution_array_conj, infodict_conj, ier_conj, msg_conj = opt.fsolve(
                        my_fun, solution_array_conj_guess, par, xtol=10*tol_is_sol, full_output=full_output)
                    # evaluationg the value of my_fun at the obtained relaxed solution
                    is_sol_conj = abs(complex(*my_fun(solution_array_conj, par)))
                    sol_conj = True
                    if np.isclose(is_sol_conj, 0, atol=tol_is_sol):
                        break
                    solution_array_conj_guess = solution_array_conj
        temp_guess = solution_array
    
    if sol:
        solution_dict['Sol_r'].append(solution_array[0])
        solution_dict['Sol_i'].append(solution_array[1])
        solution_dict['Guess'].append(m)
        solution_dict['g(x)'].append(is_sol)
        solution_dict['ier'].append(ier)
        solution_dict['msg'].append(msg)
        solution_dict['infodict'].append(infodict)
    if sol_conj:
        solution_dict['Sol_r'].append(solution_array_conj[0])
        solution_dict['Sol_i'].append(solution_array_conj[1])
        solution_dict['Guess'].append(solution_array_conj_guess)
        solution_dict['g(x)'].append(is_sol_conj)
        solution_dict['ier'].append(ier_conj)
        solution_dict['msg'].append(msg_conj)
        solution_dict['infodict'].append(infodict_conj)

    return solution_dict
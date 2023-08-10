import numpy as np
import scipy.linalg as lina
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

default_pars = {
    'k':10,
    'D':0.1,
    'v':0.5,
    'tau':1,
    'R':0.9,
    'label':'default'
}



def find_eig(**kwargs):
    """
    This function solves the char equation in complex plane for different initial guesses.

    Parameters:
        **kwargs (dict): Optional keyword arguments that can be passed to customize behavior.
            - par (dict):
                A complete set of parameters as a dictionary.
                default_pars = {
                    'k':-10,
                    'D':0.1,
                    'v':0.5,
                    'tau':1,
                    'R':0.9,
                    'label':'default'
                }
                If par is not passed, keys may be passed separately. Absent keys will take default values.
            - guess_single (complex):
                A single initial guess, complex number. Guess range will be ignored if this is passed.
            - guess_range_real ([float, float, int]):
                A list to create linspace over real axis, with the syntax [start, end, count]
            - guess_range_imag ([float, float, int]):
                A list to create linspace over imag axis, with the syntax [start, end, count]
            - tol_1 (float): tolerance to stop outer fsolve loop (passed directly to fsolve).
            - tol_2 (float): tolerance to stop inner fsolve loop (passed directly to fsolve).
            - tol_multiplier (float): to relax inner loop tolerance when saving the result.            

    Returns:
        pd.DataFrame: DataFrame of the solutions found with each solution in a row, containing the following columns:
            'Sol_r':    Real part of the obtained solution
            'Sol_i':    Imag part of the obtained solution
            'Guess_r':  Real part for the initial guess resulting in the obtained solution
            'Guess_i':  Imag part for the initial guess resulting in the obtained solution
            'g(x)':     Char equation value of the obtained solution
            'Label':    Label of the parameters leading to the obtained solution
            'Pars':     Complete pars dictionary leading to the obtained solution
            'kwargs':   Complete kwargs dictionary leading to the obtained solution
            the dataframe is sorted by 'Sol_r' column.
    """
    # Assign default values to missing keyword arguments for parameters
    if not 'par' in kwargs:
        par = default_pars.copy()
        for key in par:
            if key != 'label':
                par[key] = kwargs.get(key, par[key])
        if par != default_pars:
            par['label'] = kwargs.get('label', 'no_label')
    else:
        par = kwargs['par']
        
    # Assign default values to missing keyword arguments for initial guess values
    if not 'guess_single' in kwargs:
        guess_range_real = kwargs.get('guess_range_real', [-100,-100,1])
        guess_range_imag = kwargs.get('guess_range_imag', [5,5,1])
    else:
        guess_single_r = np.real(kwargs['guess_single'])
        guess_single_i = np.imag(kwargs['guess_single'])

        guess_range_real = [guess_single_r, guess_single_r, 1]
        guess_range_imag = [guess_single_i, guess_single_i, 1]
    
    # Assign default values to the rest of missing keyword arguments
    tol_1 = kwargs.get('tol_1', 1e-6)
    tol_2 = kwargs.get('tol_2', 1e-12)

    tol_multiplier = kwargs.get('tol_multiplier', 100)

    # Constructiong a dictionary to capture legit solutions
    solution_dict = {'Sol_r':[],'Sol_i':[],'Guess_r':[],'Guess_i':[],'g(x)':[], 'Label':[], 'Pars':[], 'kwargs':[]}

    # Constructiong a 2D mesh for different initial guess values
    mesh_builder = np.meshgrid(np.linspace(guess_range_real[0],guess_range_real[1],guess_range_real[2]),np.linspace(guess_range_imag[0],guess_range_imag[1],guess_range_imag[2]))
    mesh = mesh_builder[0] + mesh_builder[1] * 1j
    
    def char_eq(x):
        """
        This function evaluates the charachteristic equation at a given point.

        Parameters:
            x ([float, float]):
                A list of 2 elements, making up the complex eigenvalue to calculate char_eq.
        
        Returns:
            array[float, float]:
                A list of 2 elements, making up the complex value of char_eq at the given point.
        """
        l = complex(x[0], x[1])
        A = np.array([
            [0, 1, 0],
            [(l- par['k'])/par['D'], par['v']/par['D'], 0],
            [0, 0, par['tau'] * l]
        ])
        Q = lina.expm(A)
        q = np.insert(Q,0,0)
        y = par['D'] * q[4] * q[9] + par['v'] * (q[5] * q[9] + par['R'] * (q[1] + q[5] - q[2] * q[4]))
        return np.array([y.real, y.imag])

    for i in mesh:
        for m in i:
            m = np.array([m.real, m.imag])                      # obtaining an initial guess from the mesh as a complex number
            solution_array_initial = opt.fsolve(char_eq,m,xtol=tol_1)   # solving char_eq with a relaxed tol
            is_sol_initial = char_eq(solution_array_initial)                   # evaluationg the value of char_eq at the obtained relaxed solution
            is_sol_initial = (abs(complex(is_sol_initial[0],is_sol_initial[1])))
            # An inner loop seems to be necessary as sometimes the fsolve gives incorrect results that are ~+-2*pi from the radial complex answer of the real solution
            if is_sol_initial < tol_1 * tol_multiplier:
                solution_array = opt.fsolve(char_eq,solution_array_initial,xtol=tol_2)
                is_sol = char_eq(solution_array)                   # evaluationg the value of char_eq at the obtained relaxed solution
                is_sol = (abs(complex(is_sol[0],is_sol[1])))
                if np.isclose(is_sol,0,atol=tol_2*tol_multiplier):
                    solution_dict['Guess_r'].append(m[0])
                    solution_dict['Guess_i'].append(m[1])
                    solution_dict['g(x)'].append(is_sol)
                    solution_dict['Sol_r'].append(solution_array[0])
                    solution_dict['Sol_i'].append(solution_array[1])
                    solution_dict['Label'].append(par['label'])
                    solution_dict['Pars'].append(par)
                    solution_dict['kwargs'].append(kwargs)
                    solution_dict['Guess_r'].append(m[0])
                    solution_dict['Guess_i'].append(-m[1])
                    solution_dict['g(x)'].append(is_sol)
                    solution_dict['Sol_r'].append(solution_array[0])
                    solution_dict['Sol_i'].append(-solution_array[1])
                    solution_dict['Label'].append(par['label'])
                    solution_dict['Pars'].append(par)
                    solution_dict['kwargs'].append(kwargs)
                    continue

    solution_df = pd.DataFrame(solution_dict)
    solution_df = solution_df.sort_values(by=['Sol_r'])
    
    return solution_df


pars_list_k = []
pars_list_D = []
pars_list_tau = []
pars_list_R = []

par_list_list = [pars_list_k, pars_list_D, pars_list_tau, pars_list_R]
par_key_list = ['k', 'D', 'tau', 'R']

for i in np.linspace(0.8,1.2,5):
    for j in range(4):
        par = default_pars.copy()
        par[par_key_list[j]] = par[par_key_list[j]] * i
        par_list_list[j].append(par)

R = [-300,50,700]
I = [0,25,50]

c = 0
# for P in par_list_list:
#     for p in P:
#         c += 1
#         df = find_eig(par=p, guess_range_real=R, guess_range_imag=I)
#         filename = str(c) + ".csv"
#         df.to_csv(filename)

def plot_eig(results, ax_xlim=[-300,25], ax_ylim=[-20,20]):
    n = len(results)
    row = int(np.ceil(np.sqrt(n)*1.2))
    col = int(np.ceil(n/row))
    plt.ioff()
    fig, axes = plt.subplots(nrows=row, ncols=col)
    for i in range(n):
        Result = results[i]
        # df = Result.df
        title = []
        # for a in par_attributes:
        #     if getattr(Result.pars,a) != getattr(default_pars,a):
        #         title.append(f'{a}={getattr(Result.pars,a)} (was {getattr(default_pars,a)}). ')
        if title == []:
            title = ['Default: ']
            # for a in par_attributes:
            #     title.append(f'{a}={getattr(default_pars,a)}; ')
        # if axes.ndim == 1:
        #     df.plot.scatter(x='Sol_r',y='Sol_i',ax=axes[i])
        #     axes[i].set_title(title)
        #     axes[i].axhline(y=0, color='k')
        #     axes[i].axvline(x=0, color='k')
        #     axes[i].set_xlim(ax_xlim[0],ax_xlim[1])
        #     axes[i].set_ylim(ax_ylim[0],ax_ylim[1])
        # else:
        #     r,c = divmod(i,col)
        #     df.plot.scatter(x='Sol_r',y='Sol_i',ax=axes[r,c])
        #     axes[r,c].set_title(title)
        #     axes[r,c].axhline(y=0, color='k')
        #     axes[r,c].axvline(x=0, color='k')
        #     axes[r,c].set_xlim(ax_xlim[0],ax_xlim[1])
        #     axes[r,c].set_ylim(ax_ylim[0],ax_ylim[1])
    plt.show()

r = pd.read_csv('1.csv')

r.plot(x='Sol_r', y='Sol_i', kind='scatter')
plt.show()
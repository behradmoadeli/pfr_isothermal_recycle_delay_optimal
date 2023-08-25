def par_enhance(par, **kwargs):
    """
    Enhances a parameter dictionary by assigning default values and generating labels for parameters.

    Args:
        par (dict): A dictionary containing parameter values. Keys correspond to parameter names,
                    and values correspond to parameter values.

        default_pars (dict, optional): A dictionary containing default parameter values. If not provided,
                                       it attempts to load default parameters from 'pars_list.csv'.

        **kwargs: Additional keyword arguments that correspond to single parameter values to be updated:

            - default_pars (dict, optional): A dictionary containing default parameter values. If not provided,
                                             it attempts to load default parameters from 'pars_list.csv'.
            - pars_list (str): Path to a .csv file containing default parameter


    Returns:
        dict: An enhanced parameter dictionary with default values assigned to missing parameters
              and a label generated based on differing parameter values.
    """
    
    import numpy as np
    from .create_custom_pars_list import create_custom_pars_list
    
    # Load default parameters if not provided
    if 'default_pars' in kwargs:
        default_pars = kwargs['default_pars']
    else:
        default_pars = None
    
    if not default_pars:
        try:
            default_pars = create_custom_pars_list(kwargs.get('pars_list', 'pars_list.csv'))[1]
        except:
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
    if par['label'] == '':
        label_needed = True
    new_par = par.copy()
    for key in par:
        new_par[key] = kwargs.get(key, par[key])
    if par != new_par:
        label_needed = True

    # Creating a label for parameters if needed
    if label_needed:
        differing_pairs = {}
        for key, value in new_par.items():
            if key != 'label':
                if not np.isclose(value, default_pars[key]):
                    differing_pairs[key] = value
        new_par['label'] = '_'.join(
            [f"({key}_{value:.3g})" for key, value in differing_pairs.items()])

    return new_par
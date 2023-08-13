import numpy as np
import scipy.linalg as lina
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import os # Flexible loading and saving dataframs from/to csv files
import ast # Module to parse string representations of dictionaries
import hdbscan # For clustering solution dataframes
from sklearn.preprocessing import StandardScaler # Used in clustering function
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
        **kwargs (keyword arguments):            
            - par (dict): A dictionary containing parameters for the system's matrix. If not provided, keys may be passed separately. Absent keys will take default values.
            - guess_single (complex): A single initial guess for eigenvalue calculation (real + imaginary part).
            - guess_range_real (list): A list specifying the range of real parts of initial guess values.
            - guess_range_imag (list): A list specifying the range of imaginary parts of initial guess values.
            - tol_1 (float): Tolerance for initial guess refinement.
            - tol_2 (float): Tolerance for final eigenvalue precision.
            - tol_multiplier (float): Multiplier for refining tolerance in the final solution.
            
    Returns:
        tuple:
            A tuple containing:
            
            - solution_df (pandas.DataFrame): DataFrame containing found solutions' information.
            - label (str): A label describing the customized parameters used for the computation.
            - metadata (dict): A dictionary containing input parameter values used in the computation.
    """
    
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
        differing_pairs = {key: value for key, value in par.items() if not np.isclose(value, default_pars[key])}
        par['label'] = '_'.join([f"({key}_{value:.3g})" for key, value in differing_pairs.items()])
            
    # Assign default values to missing keyword arguments for initial guess values
    if 'guess_single' in kwargs:
        guess_single_r = np.real(kwargs['guess_single'])
        guess_single_i = np.imag(kwargs['guess_single'])

        guess_range_real = [guess_single_r, guess_single_r, 1]
        guess_range_imag = [guess_single_i, guess_single_i, 1]
    else:
        guess_range_real = kwargs.get('guess_range_real', [-100,-100,1])
        guess_range_imag = kwargs.get('guess_range_imag', [5,5,1])
    
    # Assign default values to the rest of missing keyword arguments
    tol_1 = kwargs.get('tol_1', 1e-4)
    tol_2 = kwargs.get('tol_2', 1e-15)
    tol_multiplier = kwargs.get('tol_multiplier', 100)

    metadata = {
        'par' : par,
        'guess_range' : (guess_range_real, guess_range_imag),
        'tols' : (tol_1, tol_2, tol_multiplier)
    }

    # Constructiong a dictionary to capture legit solutions
    solution_dict = {
        'Sol_r':[],'Sol_i':[],'Guess_r':[],'Guess_i':[],'g(x)':[]
        }

    # Constructiong a 2D (Re-Im plane) mesh for different initial guess values
    mesh_builder = np.meshgrid(
        np.linspace(guess_range_real[0],guess_range_real[1],guess_range_real[2]),
        np.linspace(guess_range_imag[0],guess_range_imag[1],guess_range_imag[2])
        )
    mesh = mesh_builder[0] + mesh_builder[1] * 1j
    
    def char_eq(x): # To be more simplified
        """
        This function evaluates the charachteristic equation at a given point.

        Parameters:
            x ([float, float]):
                A list of 2 elements, making up the Re and Im parts of the complex eigenvalue to calculate char_eq.
        
        Returns:
            array[float, float]:
                An array of 2 elements, making up the Re and Im parts of the complex value of char_eq at the given x.
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
            m = np.array([m.real, m.imag]) # obtaining an initial guess from the mesh as a complex number
            solution_array_initial = opt.fsolve(char_eq,m,xtol=tol_1) # solving char_eq with a relaxed tol
            is_sol_initial = char_eq(solution_array_initial) # evaluationg the value of char_eq at the obtained relaxed solution
            is_sol_initial = (abs(complex(is_sol_initial[0],is_sol_initial[1])))
            # An inner loop seems to be necessary as sometimes the fsolve gives incorrect results that are ~+-2*pi from the radial complex answer of the real solution
            if np.isclose(is_sol_initial,0,atol=tol_1*tol_multiplier):
                solution_array = opt.fsolve(char_eq,solution_array_initial,xtol=tol_2)
                is_sol = char_eq(solution_array) # evaluationg the value of char_eq at the obtained relaxed solution
                is_sol = (abs(complex(is_sol[0],is_sol[1])))
                if np.isclose(is_sol,0,atol=tol_2*tol_multiplier):
                    solution_dict['Guess_r'].append(m[0])
                    solution_dict['Guess_i'].append(m[1])
                    solution_dict['g(x)'].append(is_sol)
                    solution_dict['Sol_r'].append(solution_array[0])
                    solution_dict['Sol_i'].append(solution_array[1])
                    solution_array_conj_guess = solution_array.copy()
                    solution_array_conj_guess[1] *= -1
                    solution_array_conj = opt.fsolve(char_eq,solution_array_conj_guess,xtol=tol_2)
                    is_sol_conj = char_eq(solution_array_conj) # evaluationg the value of char_eq at the obtained relaxed solution
                    is_sol_conj = (abs(complex(is_sol_conj[0],is_sol_conj[1])))
                    if np.isclose(is_sol_conj,0,atol=tol_2*tol_multiplier):
                        solution_dict['Guess_r'].append(m[0])
                        solution_dict['Guess_i'].append(-m[1])
                        solution_dict['g(x)'].append(is_sol_conj)
                        solution_dict['Sol_r'].append(solution_array_conj[0])
                        solution_dict['Sol_i'].append(solution_array_conj[1])
    
    solution_df = pd.DataFrame(solution_dict)
    solution_df = solution_df.sort_values(by=['Sol_r'])
    
    return (solution_df, par['label'], metadata)

def save_dataframe_to_csv(df, filename, parent_dir=None, metadata=None):
    """
    Save a DataFrame to a CSV file, optionally with metadata in the first line.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        filename (str): The name of the output CSV file.
        parent_dir (str, optional): The parent directory to save the file in. If None, saves in the current directory.
        metadata (str, optional): Metadata to be stored in the first line of the CSV file as a comment.

    Returns:
        None
    """
    if parent_dir:
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        output_filepath = os.path.join(parent_dir, filename)
    else:
        output_filepath = filename
    
    counter = 1
    while os.path.exists(output_filepath):
        # If the file already exists, generate a new filename with an incremental suffix
        base_name, ext = os.path.splitext(filename)
        output_filepath = os.path.join(parent_dir, f"{base_name}_{counter}{ext}")
        counter += 1
        
    if metadata is None:
        df.to_csv(output_filepath, index=False)
        print(f"DataFrame saved to {output_filepath}")
    else:
        with open(output_filepath, 'w') as f:
            f.write(f"# {metadata}\n")
            df.to_csv(f, index=False)
        print(f"DataFrame with metadata '{metadata}' saved to {output_filepath}")

def load_dataframe_from_csv(input_filepath):
    """
    Load a DataFrame from a CSV file, along with metadata if present in the first line.

    Args:
        input_filepath (str): The path to the input CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
        str: The base filename (without extension) of the input file.
        str or None: Metadata if present, else None.
    """
    with open(input_filepath, 'r') as f:
        metadata_line = f.readline().strip()
        if metadata_line.startswith('#'):
            metadata = metadata_line.lstrip('# ').strip()
            df = pd.read_csv(input_filepath, comment='#')
        else:
            metadata = None
            df = pd.read_csv(input_filepath)
    
    base_filename = os.path.splitext(os.path.basename(input_filepath))[0]
    return df, base_filename, metadata


def create_custom_pars_list(csv_path):
    """
    Create a list of custom parameter dictionaries from a CSV file.

    This function reads a CSV file located at the given path and constructs a list
    of dictionaries, where each dictionary represents custom parameters based on
    the CSV content and a global default_pars dictionary.

    Parameters:
        csv_path (str): The path to the CSV file.

    Returns:
        list: A list of dictionaries containing custom parameter sets.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Initialize a list to store dictionaries
    custom_pars_list = []
    
    # Use the global default_pars variable
    global default_pars

    # Iterate through the rows of the DataFrame
    for index, row in df.iterrows():
        custom_pars = default_pars.copy()
        
        # Update custom_pars with values from the current row
        for key in default_pars.keys():
            if key in row:
                custom_pars[key] = row[key]
        
        # Append the customized parameter dictionary to the list
        custom_pars_list.append(custom_pars)
    
    return custom_pars_list

def cluster(input_df, min_cluster_size):
    """
    Generate a representative DataFrame from an input DataFrame using HDBSCAN clustering.

    Parameters:
        input_df (pd.DataFrame): The input DataFrame with 'x' and 'y' columns.
        min_cluster_size (int): Minimum number of points to form a cluster (default is 2).

    Returns:
        pd.DataFrame: A DataFrame containing one representative point from each cluster.
    """
    # Standardize the data
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(input_df[['Sol_r', 'Sol_i']])
    
    # Apply HDBSCAN clustering
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = hdbscan_clusterer.fit_predict(points_scaled)
    
    # Create a new DataFrame with one representative point from each cluster
    unique_labels = set(labels)
    representative_points = []

    for label in unique_labels:
        if label != -1:
            cluster_points = input_df[labels == label][['x', 'y']]
            representative_point = cluster_points.iloc[0]
            representative_points.append(representative_point)

    representative_df = pd.DataFrame(representative_points, columns=['x', 'y'])
    return representative_df

def plot_single_df(
        df, title, metadata, filter=True,
        real_lower_bound=-200, real_upper_bound=200, imag_lower_bound=-5, imag_upper_bound=5
        ):
    """
    Plot a DataFrame based on several inputs.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        title (str): Title for the plot.
        metadata (dict): Dictionary containing metadata information.
        filter_real (float): Minimum value for real part.
        filter_imag (float): Absolute maximum value for imaginary part.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if filter:
        # Filter rows based on criteria
        filtered_df = df[
            (df['Sol_r'] > real_lower_bound) &
            (df['Sol_r'] < real_upper_bound) &
            (df['Sol_i'] > imag_lower_bound) &
            (df['Sol_i'] < imag_upper_bound)
        ]
    
    # Scatter plot
    ax.scatter(filtered_df['Sol_r'], filtered_df['Sol_i'])
    ax.set_title(title)
    ax.set_xlabel('Sol_r')
    ax.set_ylabel('Sol_i')
    
    # Set axis limits to ensure visibility of x=0 and y=0 lines
    ax.axhline(0, color='black', linewidth=0.5)  # Horizontal line at y=0
    ax.axvline(0, color='black', linewidth=0.5)  # Vertical line at x=0
    ax.grid(True)
    plt.show()
    
    # Print metadata
    for key, value in metadata.items():
        print(f'{key} : {value}')

# Solve and plot for default_par

df, label, metadata = find_eig(guess_range_real=[-200,50,25], guess_range_imag=[0,10,5])
df = cluster(df, 5)
plot_single_df(df, label, metadata)
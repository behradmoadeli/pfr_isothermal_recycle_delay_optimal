# Loading the packages:
import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import os  # Flexible loading and saving dataframs from/to csv files
import hdbscan  # For clustering solution dataframes
from sklearn.preprocessing import StandardScaler  # Used in clustering function
import warnings
warnings.simplefilter('ignore')

def generate_unique_file_path(file_path):
    """
    Generate a unique file path by appending a counter to the filename if the file already exists.

    Args:
        file_path (str): The desired file path.

    Returns:
        str: A unique file path that doesn't already exist.
    """
    base_path, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = file_path
    while os.path.exists(new_file_path):
        new_file_path = f"{base_path}_{counter}{ext}"
        counter += 1
    return new_file_path


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
    # Ensure the filename ends with '.csv'
    if not filename.endswith('.csv'):
        filename += '.csv'

    if parent_dir:
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        output_filepath = os.path.join(parent_dir, filename)
    else:
        output_filepath = filename

    output_filepath = generate_unique_file_path(output_filepath) # Prevents overwriting an existing file

    if metadata is None:
        df.to_csv(output_filepath, index=False)
        print(f"DataFrame saved to {output_filepath}")
    else:
        with open(output_filepath, 'w') as f:
            f.write(f"# {metadata}\n")
            df.to_csv(f, index=False)
        print(
            f"DataFrame with metadata '{metadata}' saved to {output_filepath}")


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
    df = pd.read_csv(csv_path).fillna('')

    # Initialize a list to store dictionaries
    custom_pars_list = []

    # Use the global default_pars variable
    global default_pars
    default_pars = df[df['label'] == 'default'].to_dict(orient='records')[0] if any(df['label'] == 'default') else default_pars

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


def significant_digits(x, n):
    if x == 0:
        return 0
    return n - int(np.floor(np.log10(abs(x)))) - 1


def process_dataframe(df, n):
    """
    Process a pandas DataFrame by adding an 'instances' column, rounding values,
    and handling duplicates. Also replaces small 'Sol_i' values with zero.

    Args:
        df (pandas.DataFrame): The input DataFrame with 'Sol_r' and 'Sol_i' columns.

    Returns:
        pandas.DataFrame: Processed DataFrame with 'instances' column updated.
    """
    # Add a new column 'instances' with default value 1
    df['instances'] = 1

    # Create a dictionary to keep track of unique rounded values
    unique_values = {}

    # Iterate through rows of the dataframe
    for index, row in df.iterrows():
        # Replace small 'Sol_i' values with zero
        sol_i = row['Sol_i']
        if sol_i < 1e-8:
            sol_i = 0

        # Round the values to 5 significant digits
        rounded_r = round(row['Sol_r'], significant_digits(row['Sol_r'], n))
        rounded_i = round(sol_i, significant_digits(sol_i, n))

        # Check if the rounded values have been encountered before
        rounded_key = (rounded_r, rounded_i)
        if rounded_key in unique_values:
            # Increment 'instances' of the existing row
            existing_index = unique_values[rounded_key]
            df.at[existing_index, 'instances'] += 1
            df.drop(index, inplace=True)  # Remove the current row
        else:
            unique_values[rounded_key] = index

    return df


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
            cluster_points = input_df[labels == label][['Sol_r', 'Sol_i']]
            representative_point = cluster_points.iloc[0]
            representative_points.append(representative_point)

    representative_df = pd.DataFrame(
        representative_points, columns=['Sol_r', 'Sol_i'])
    return representative_df


def plot_single_df(
        df, title, metadata, filter=True,
        real_lower_bound=-200, real_upper_bound=50, imag_lower_bound=-5, imag_upper_bound=5
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
    else:
        filtered_df = df

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


def plot_multiple_datasets(
        parent_directory, save_path=None,
        filter=False, filter_range=[(-200, 10), 100],
        rows=None, cols=None, figsize=None, **kwargs):
    """
    Plot multiple datasets from CSV files in a grid layout.

    Args:
        parent_directory (str): The directory containing CSV files to plot.
        rows (int, optional): Number of rows in the grid. If None, calculated based on dataset count.
        cols (int, optional): Number of columns in the grid. If None, calculated based on dataset count.
        figsize (tuple, optional): Figure size (width, height) in inches.
        **kwargs: Additional keyword arguments to pass to matplotlib.subplots().

    Returns:
        None
    """
    # List all CSV files in the parent directory
    csv_files = [file for file in os.listdir(
        parent_directory) if file.endswith('.csv')]
    results = []

    # Load dataframes from CSV files and categorize into 'default' and other datasets
    for csv_file in csv_files:
        csv_path = os.path.join(parent_directory, csv_file)
        df, base_filename, metadata = load_dataframe_from_csv(csv_path)
        if base_filename == 'default':
            default = df
        else:
            results.append((df, base_filename, metadata))

    sort_sequence = ['v', 'D', 'k', 't', 'R']

    # Define a sorting function that extracts the first alphabetic letter from the second element of each tuple
    def custom_sort_key(item):
        second_element = str(item[1])  # Convert the second element to a string
        first_alpha_letter = next(
            (char for char in second_element if char.isalpha()), '')
        return sort_sequence.index(first_alpha_letter) if first_alpha_letter in sort_sequence else len(sort_sequence)

    # Sort the results list based on the custom sorting key
    results = sorted(results, key=custom_sort_key)

    # Extract datasets and calculate the number of datasets
    datasets = [result[0] for result in results]
    labels = [result[1] for result in results]
    num_datasets = len(datasets)

    # Calculate rows and columns for the grid layout if not specified
    if rows is None and cols is None:
        rows = int(np.ceil(np.sqrt(num_datasets)))
        cols = int(np.ceil(num_datasets / rows))
    elif rows is None:
        rows = int(np.ceil(num_datasets / cols))
    elif cols is None:
        cols = int(np.ceil(num_datasets / rows))

    mid_row = int(np.floor(rows/2))

    # Set default figure size if not specified
    if figsize is None:
        figsize = (cols * 8, rows * 8)

    # Create the subplots grid
    fig, axes = plt.subplots(rows+1, cols, figsize=figsize, **kwargs)

    # Flatten the axes array if there's only one dataset
    if num_datasets == 1:
        axes = np.array([[axes]])

    if filter:
        # Filter rows based on criteria
        default = default[
            (default['Sol_r'] > min(filter_range[0])) &
            (default['Sol_r'] < max(filter_range[0])) &
            (default['Sol_i'] > -abs(filter_range[1])) &
            (default['Sol_i'] < abs(filter_range[1]))
        ]

    # Iterate through datasets and plot them on the grid
    for i, dataset in enumerate(datasets):

        if filter:
            # Filter rows based on criteria
            dataset = dataset[
                (dataset['Sol_r'] > min(filter_range[0])) &
                (dataset['Sol_r'] < max(filter_range[0])) &
                (dataset['Sol_i'] > -abs(filter_range[1])) &
                (dataset['Sol_i'] < abs(filter_range[1]))
            ]

        row = i % rows  # Calculate current row index
        col = i // rows  # Calculate current column index

        # Plot default dataset
        if row == mid_row:
            # Plot each dataset
            axes[row, col].scatter(default['Sol_r'], default['Sol_i'])
            axes[row, col].set_title('Default')

            # Add gridlines
            axes[row, col].grid(True)

            # Show both x-axis and y-axis lines
            # Horizontal line at y=0
            axes[row, col].axhline(0, color='black', linewidth=0.8)
            # Vertical line at x=0
            axes[row, col].axvline(0, color='black', linewidth=0.8)

            row += 1

        elif row >= mid_row:
            row += 1

        # Plot each dataset
        axes[row, col].scatter(dataset['Sol_r'], dataset['Sol_i'])
        axes[row, col].set_title(labels[i])

        # Add gridlines
        axes[row, col].grid(True)

        # Show both x-axis and y-axis lines
        # Horizontal line at y=0
        axes[row, col].axhline(0, color='black', linewidth=0.8)
        # Vertical line at x=0
        axes[row, col].axvline(0, color='black', linewidth=0.8)

    # Remove empty subplots if there are more grids than datasets
    for i, ax in enumerate(axes.flat):
        if not ax.get_title():
            ax.axis('off')

    # Adjust layout and display the plot
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        if not save_path.endswith('.png'):
            save_path += '.png'
        save_path = generate_unique_file_path(save_path)
        plt.savefig(save_path)  # Save the figure as an image file
        plt.close()  # Close the figure to release resources


def char_eq(x):  # To be more simplified
    """
    This function evaluates the charachteristic equation at a given point.

    Parameters:
        x ([float, float]):
            A list of 2 elements, making up the Re and Im parts of the complex eigenvalue to calculate char_eq.

    Returns:
        array[float, float]:
            An array of 2 elements, making up the Re and Im parts of the complex value of char_eq at the given x.
    """
    global global_par  # To access 'global_par', defined within 'find_eig()' function as a global variable.
    par = global_par
    l = complex(x[0], x[1])

    (k, v, D, t, R) = (par['k'], par['v'], par['D'], par['tau'], par['R'])
    p = v**2 - 4*D * (k-l)
    p_sqrt = np.sqrt(p)
    if np.isclose(p_sqrt, 0, atol=1e-8):
        y = (
            np.exp(l*t+v/2/D) * (v**2 + 2*D**2)
            + v * (1-R*np.exp(v/D))
        )
    else:
        y = (
            np.exp(l*t+v/2/D) * np.sinh(np.sqrt(p)/2/D) * (v**2 + 2*D**2)
            + v * np.sqrt(p) * (np.cosh(np.sqrt(p)/2/D)-R*np.exp(v/D))
        )

    return np.array([y.real, y.imag])


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

    global global_par  # Store 'par' as a global variable
    # So that it can be accessed by char_eq(x), without being passed to it
    global_par = par

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
            solution_array = opt.fsolve(char_eq, m, xtol=tol_fsolve)
            # evaluationg the value of char_eq at the obtained relaxed solution
            is_sol = char_eq(solution_array)
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
                    char_eq, solution_array_conj_guess, xtol=tol_fsolve)
                # evaluationg the value of char_eq at the obtained relaxed solution
                is_sol_conj = char_eq(solution_array_conj)
                is_sol_conj = (abs(complex(is_sol_conj[0], is_sol_conj[1])))
                if np.isclose(is_sol_conj, 0, atol=tol_is_sol):
                    solution_dict['Guess_r'].append(m[0])
                    solution_dict['Guess_i'].append(-m[1])
                    solution_dict['g(x)'].append(is_sol_conj)
                    solution_dict['Sol_r'].append(solution_array_conj[0])
                    solution_dict['Sol_i'].append(solution_array_conj[1])

    del global_par  # Avoid storing a global parameter after making use of it

    solution_df = process_dataframe(
        pd.DataFrame(solution_dict), round_sig_digits)
    solution_df = solution_df.sort_values(by=['Sol_r'], ascending=False)
    return (solution_df, par['label'], metadata)


# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script's directory
os.chdir(script_directory)

# Now the CWD is the same as the script's directory
print("New Current Working Directory:", os.getcwd())

# Initializing system parameters:

default_pars = {
    'k': 10,
    'D': 0.1,
    'v': 0.5,
    'tau': 1,
    'R': 0.9,
    'label': 'default'
}

# for par in pars_list:
#     df, label, metadata = find_eig(par=par)
#     save_dataframe_to_csv(df, label, 'CSV', metadata)


plot_multiple_datasets('CSV', save_path='multi_plot', filter=True)

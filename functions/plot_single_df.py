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
    import matplotlib.pyplot as plt
    import pandas as pd
    
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
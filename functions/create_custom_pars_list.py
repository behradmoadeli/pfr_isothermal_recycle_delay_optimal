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
    import pandas as pd
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path).fillna('')

    # Initialize a list to store dictionaries
    custom_pars_list = []

    default_pars = {
        'k': 10,
        'D': 0.1,
        'v': 0.5,
        'tau': 1,
        'R': 0.9,
        'label': 'default'
    }

    default_pars = df[df['label'] == 'default'].to_dict(orient='records')[0] if any(df['label'] == 'default') else default_pars

    # Iterate through the rows of the DataFrame
    for index, row in df.iterrows():
        custom_pars = default_pars.copy()

        # Update custom_pars with values from the current row
        for key in default_pars.keys():
            if key in row:
                custom_pars[key] = row[key]

        # Append the customized parameter dictionary to the list
        if custom_pars != default_pars:
            custom_pars_list.append(custom_pars)

    return custom_pars_list, default_pars